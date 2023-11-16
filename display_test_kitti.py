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
from datasets.KittiDataset import KittiTestset
from datasets.dataloader import get_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='kitti10261814', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=500, type=int)
    args = parser.parse_args()

    config_path = f'/home/vision/ADD_prj/D3Feat.pytorch/data/kitti/snapshot/{args.chosen_snapshot}/config_kitti_map.json'
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

    model = KPFCNN(config).to('cuda')
    model.load_state_dict(torch.load(f'/home/vision/ADD_prj/D3Feat.pytorch/data/kitti/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    model.eval()

    data_timer, model_timer = Timer(), Timer()

    data_timer.tic()
    dset = KittiTestset(root=config.root,
                                downsample=config.downsample,
                                config=config,
                                )
    dloader, _ = get_dataloader(dataset=dset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0
                                )

    total_dataset_length = len(dloader.dataset)
    dataloader_iter = dloader.__iter__()
    data_timer.toc()
    print(f"data time: {data_timer.avg:.2f}s ")

    for _ in range(total_dataset_length//2):
        inputs = next(dataloader_iter)
        for k, v in inputs.items():  # load inputs to device.
            if type(v) == list:
                inputs[k] = [item.to('cuda') for item in v]
            else:
                inputs[k] = v.to('cuda')
        
        pts0 = inputs['points'][0].cpu().numpy()
        model_timer.tic()
        features, scores = model(inputs)
        model_timer.toc()
        print(f"model time: {model_timer.avg:.2f}s ")
        features_np = features.cpu().detach().numpy()[:,:3]
        scores_np = scores.cpu().detach().numpy()

        top_indices0 = np.argsort(scores_np[:, 0])[-500:]
        color0 = np.repeat(np.array([[1.0,0.8,0.8]]), len(scores_np), axis=0)
        for i in top_indices0:
            color0[i] = [1,0,0]

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
        
        ## 수동으로 조정하고 싶을 경우 사용 ##
        # # 회전 각도 (-90도를 라디안으로 표현)
        # theta = -np.pi/2  # -90도
        # theta = 0  # -90도

        # # Z축 주위의 회전 행렬 생성
        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, -0],
        #                             [np.sin(theta), np.cos(theta), 0, 0],
        #                             [0, 0, 1, 0], [0,0,0,1]])
        ################################
        rotation_matrix = np.eye(4, 4)

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

        visualizer.add_geometry(pcd_vis0)
        visualizer.add_geometry(pcd_vis1)
        visualizer.get_render_option().show_coordinate_frame = True
        visualizer.run()
        visualizer.destroy_window()

        # keypoint로 icp
        pcd_k0 = o3d.geometry.PointCloud()
        pcd_k1 = o3d.geometry.PointCloud()
        pcd_k0.points = o3d.utility.Vector3dVector(pts0[top_indices0])
        pcd_k1.points = o3d.utility.Vector3dVector(pts1[top_indices1])

        reg = o3d.pipelines.registration.registration_icp(
                pcd_k0, pcd_k1, 0.5, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100000))
        pcd_vis0.transform(reg.transformation)

        visualizer1 = o3d.visualization.Visualizer()
        visualizer1.create_window()
        visualizer1.add_geometry(pcd_vis0)
        visualizer1.add_geometry(pcd_vis1)
        visualizer1.get_render_option().show_coordinate_frame = True
        visualizer1.run()
        visualizer1.destroy_window()