import os
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud
from models.architectures import KPFCNN
from utils.timer import Timer, AverageMeter
from datasets.KittiDataset import KittiTestDatasetPKL
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence
from config import get_config
import copy


def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'kitti')
                target_desc = get_desc(descpath, cloud_bin_t, 'kitti')
                source_score = get_scores(scorepath, cloud_bin_s, 'kitti').squeeze()
                target_score = get_scores(scorepath, cloud_bin_t, 'kitti').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)
                
                # randomly select 5000 keypts
                if args.random_points:
                    source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(source_score)[-args.num_points:]
                    target_indices = np.argsort(target_score)[-args.num_points:]
                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]
                
                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    
    # generate descriptors
    recall_list = []
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"{config.root}/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()
            features, scores = model(inputs)
            pcd_size = inputs['stack_lengths'][0][0]
            pts = inputs['points'][0][:int(pcd_size)]
            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
            # scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.kiit', features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")
    

if __name__ == '__main__':
    # configt = get_config()
    # dconfig = vars(config)
    # for k in dconfig:
    #     print(f"    {k}: {dconfig[k]}")
    # config = edict(dconfig)

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='kitti_map10261814', type=str, help='snapshot dir') #>> kitti10251818
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=500, type=int)
    parser.add_argument('--generate_features', default='/media/vision/Seagate/DataSets/kitti_map/dataset/sequences', type=str)
    parser.add_argument('--dataset_root', default='/media/vision/Seagate/DataSets/kitti_map/dataset/sequences', type=str)
    parser.add_argument('--dataset_pose_root', default='/media/vision/Seagate/DataSets/kitti_map/dataset/poses', type=str)
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO, 
        filename=log_filename, 
        filemode='w', 
        format="")

# /home/vision/forD3feat/D3Feat.pytorch/data/kitti/snapshot/kitti10250004
    config_path = f'/home/vision/forD3feat/D3Feat.pytorch/data/kitti_map/snapshot/{args.chosen_snapshot}/config_kitti_map.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model 
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')

    # # dynamically load the model from snapshot
    # module_file_path = f'snapshot/{chosen_snap}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # model = module.KPFCNN(config)
    
    # if test on datasets with different scale
    # config.first_subsampling_dl = [new voxel size for first layer]
    
    # make d3feat model
    model = KPFCNN(config).to('cuda')
    model.load_state_dict(torch.load(f'/home/vision/forD3feat/D3Feat.pytorch/data/kitti_map/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    model.eval()


    # save_path = f'geometric_registration/{args.chosen_snapshot}'


    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    dset = KittiTestDatasetPKL(root=config.root,
                                split='test',
                                downsample=0.3,
                                self_augment=False,
                                num_node=config.num_node,
                                augment_noise=config.augment_noise,
                                augment_axis=config.augment_axis, 
                                augment_rotation=config.augment_rotation,
                                augment_translation=config.augment_translation,
                                config=config,
                                )
    dloader, _ = get_dataloader(dataset=dset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0
                                )
    dataloader_iter = dloader.__iter__()

    ############
    inputs = next(dataloader_iter)
    for k, v in inputs.items():  # load inputs to device.
        print(k)
        if type(v) == list:
            inputs[k] = [item.to('cuda') for item in v]
        else:
            inputs[k] = v.to('cuda')
    
    pts0 = inputs['points'][0].cpu().numpy()
    features, scores = model(inputs)
    features_np = features.cpu().detach().numpy()[:,:3]
    scores_np = scores.cpu().detach().numpy()
    scores_np_t = copy.deepcopy(scores_np)
    top_indices0 = np.argsort(scores_np_t[:, 0])[-500:]
    color0 = np.repeat(np.array([[1.0,0.8,0.8]]), len(scores_np), axis=0)
    # color0 = np.array([color0])
    # color0 = np.repeat(color0, 3, axis=1)
    print(pts0.shape, features_np.shape, scores_np.shape)


    for i in top_indices0:
        color0[i] = [1,0,0]
        # print(scores_np[i])
    #################333
    inputs = next(dataloader_iter)
    for k, v in inputs.items():  # load inputs to device.
        if type(v) == list:
            inputs[k] = [item.to('cuda') for item in v]
        else:
            inputs[k] = v.to('cuda')
    
    pts1 = inputs['points'][0].cpu().numpy()
    features, scores = model(inputs)
    features_np = features.cpu().detach().numpy()[:,:3]
    scores_np = scores.cpu().detach().numpy()
    top_indices1 = np.argsort(scores_np[:, 0])[-500:]
    color1 = np.repeat(np.array([[0.8,0.8,1.0]]), len(scores_np), axis=0)
    # color1 = np.repeat(color1, 3, axis=1)
    
    # 회전 각도 (-90도를 라디안으로 표현)
    theta = -np.pi/2  # -90도

    # Z축 주위의 회전 행렬 생성
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, -10],
    #                             [np.sin(theta), np.cos(theta), 0, 10],
    #                             [0, 0, 1, 0], [0,0,0,1]])
    
    rotation_matrix = np.array([[1, 0, 0, -10],
                                [0, 1, 0, 5],
                                [0, 0, 1, 0], 
                                [0,0,0,1]])
                                

    for i in top_indices1:
        color1[i] = [0,0,1]

    pcd_vis0 = o3d.geometry.PointCloud()
    pcd_vis0.points = o3d.utility.Vector3dVector(pts0)
    pcd_vis0.colors = o3d.utility.Vector3dVector(color0)
    pcd_vis0.transform(rotation_matrix)

    pcd_vis1 = o3d.geometry.PointCloud()
    pcd_vis1.points = o3d.utility.Vector3dVector(pts1)
    pcd_vis1.colors = o3d.utility.Vector3dVector(color1)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    # pcd0.paint_uniform_color([1, 0, 0])
    # pcd1.paint_uniform_color([0, 0, 1])
    # visualizer.add_geometry(pcd0)
    visualizer.add_geometry(pcd_vis0)
    visualizer.add_geometry(pcd_vis1)
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.run()
    visualizer.destroy_window()

    pcd_k0 = o3d.geometry.PointCloud()
    pcd_k1 = o3d.geometry.PointCloud()
    pcd_k0.points = o3d.utility.Vector3dVector(pts0[top_indices0])
    pcd_k1.points = o3d.utility.Vector3dVector(pts1[top_indices1])

    

    reg = o3d.pipelines.registration.registration_icp(
            pcd_vis0, pcd_vis1, 1.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100000))
    pcd_vis0.transform(reg.transformation)

    visualizer1 = o3d.visualization.Visualizer()
    visualizer1.create_window()
    visualizer1.add_geometry(pcd_vis0)
    visualizer1.add_geometry(pcd_vis1)
    visualizer1.get_render_option().show_coordinate_frame = True
    visualizer1.run()
    visualizer1.destroy_window()


    # generate_features(model.cuda(), dloader, config, args.chosen_snapshot)

    # # register each pair of fragments in scenes using multiprocessing.
    # scene_list = [
    #     '7-scenes-redkitchen',
    #     'sun3d-home_at-home_at_scan1_2013_jan_1',
    #     'sun3d-home_md-home_md_scan9_2012_sep_30',
    #     'sun3d-hotel_uc-scan3',
    #     'sun3d-hotel_umd-maryland_hotel1',
    #     'sun3d-hotel_umd-maryland_hotel3',
    #     'sun3d-mit_76_studyroom-76-1studyroom2',
    #     'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    # ]
    # return_dict = Manager().dict()
    # # register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    # jobs = []
    # for scene in scene_list:
    #     p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene))
    #     jobs.append(p)
    #     p.start()
    
    # for proc in jobs:
    #     proc.join()

    # recalls = [v[0] for k, v in return_dict.items()]
    # inlier_nums = [v[1] for k, v in return_dict.items()]
    # inlier_ratios = [v[2] for k, v in return_dict.items()]

    # logging.info("*" * 40)
    # logging.info(recalls)
    # logging.info(f"All 8 scene, average recall: {np.mean(recalls):.2f}%")
    # logging.info(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    # logging.info(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios)*100:.2f}%")
