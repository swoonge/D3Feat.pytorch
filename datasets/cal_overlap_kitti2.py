import os
from os.path import exists, join
import pickle
import numpy as np
import open3d
import cv2
import time
import copy

def odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    T_w_cam0 = np.vstack((T_w_cam0, [0.0, 0.0, 0.0, 1.0]))
    return T_w_cam0

class Kitti_cal_overlap(object):
    """
    Given point cloud fragments and corresponding pose in '{root}'.
        1. Save the aligned point cloud pts in '{savepath}/3DMatch_{downsample}_points.pkl'
        2. Calculate the overlap ratio and save in '{savepath}/3DMatch_{downsample}_overlap.pkl'
        3. Save the ids of anchor keypoints and positive keypoints in '{savepath}/3DMatch_{downsample}_keypts.pkl'
    """

    # __init__ 메서드: 이 메서드는 클래스의 초기화를 처리합니다. 데이터셋의 루트 디렉토리 (root), 결과를 저장할 경로 (savepath), 
    # 데이터셋 분할 (split), 다운샘플링 크기 (downsample)를 인수로 받습니다. 이 메서드에서는 필요한 초기화 작업을 수행하고 데이터셋을 불러오는 데 사용됩니다.
    def __init__(self, root, pose_root, savepath, split, downsample):
        self.root = root
        self.pose_root = pose_root
        self.savepath = savepath
        self.split = split
        self.downsample = downsample
        self.poses = {}
        self.scan_gap = 10
        self.poses_pair = []
        self.marge_range = 2

        # dict: from id to pts.
        self.pts = {}
        # dict: from id_id to overlap_ratio
        self.overlap_ratio = {}
        # dict: from id_id to anc_keypts id & pos_keypts id
        self.keypts_pairs = {}

        # seq 목록 불러오기
        # if self.split == 'train': self.scene_list = ["00"]
        if self.split == 'train': self.scene_list = ["00","01","02","03","04","05"]
        elif self.split == 'val': self.scene_list = ["06","07"] 
        else: self.scene_list = ["08","09","10"] 

        self.ids_list = []
        self.ids_list_count = {}
        self.scene_to_ids = {} # 결국 이거 얻기 위함
        
        for scene in self.scene_list:
            self.ids_list_count[scene] = len(self.ids_list)
            self.get_odometry(scene)
            # bin 불러오기 위한 경로 집합 생성
            self.scene_to_ids[scene] = []
            scene_path = os.path.join(self.root, scene + f'/velodyne')
            
            # ids: 모든 bin file경로 모음
            ids = [scene + f"/velodyne/" + str(filename.split(".")[0]) for filename in os.listdir(scene_path) if filename.endswith('bin')]
            ids = sorted(ids, key=lambda x: int(x.split("/")[-1]))
            # print(ids) >> 05/velodyne/002753 같이 모든 bin file의 root 이후 경로 저장

            # self.ids_list: 모든 bin file 경로 모음
            self.ids_list += ids
            # self.scene_to_ids: 각 씬 넘버 "00" key에 bin file 경로
            self.scene_to_ids[scene] += ids

            print(f"Seq {scene}: num bin: {len(ids)}")
        print(f"Total {len(self.scene_list)} scenes, {len(self.ids_list)} point cloud fragments.")
        self.load_all_ply(downsample)
        self.cal_overlap(downsample)

    def get_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.pose_root + '/' + drive + '.txt'
        poses = np.loadtxt(data_path)
        poses = poses.reshape([-1,3,4])
        poses_matrix = np.zeros((poses.shape[0] , 4, 4))
        for i, pose in enumerate(poses): 
            poses_matrix[i] = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
        self.poses[drive] = poses_matrix
            
    # load_ply 메서드: 이 메서드는 주어진 디렉토리에서 .ply 포맷의 3D 포인트 클라우드 데이터를 로드합니다. 
    # 다운샘플링을 수행하고, 포즈 정보를 사용하여 포인트 클라우드를 정렬합니다.
    # 여기서 할꺼 다 해보자.
    def load_ply(self, fname0, fname1, drive, i, j, downsample, aligned=True): #join(data_dir, f'{ind}.bin')
        pcd0_list = []
        pcd1_list = []
        for fname in fname0:
            xyzr0 = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
            xyz0 = xyzr0[:, :3]
            pcd0 = open3d.geometry.PointCloud()
            pcd0.points = open3d.utility.Vector3dVector(xyz0)
            # pcd0, _ = pcd0.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            pcd0 = pcd0.voxel_down_sample(voxel_size=0.1)
            pcd0_list.append(pcd0)

        for fname in fname1:
            xyzr1 = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
            xyz1 = xyzr1[:, :3]
            pcd1 = open3d.geometry.PointCloud()
            pcd1.points = open3d.utility.Vector3dVector(xyz1)
            # pcd1, _ = pcd1.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            pcd1 = pcd1.voxel_down_sample(voxel_size=0.1)
            pcd1_list.append(pcd1)

        pcd0_list_aligned = [pcd0_list[0]]
        pcd1_list_aligned = [pcd1_list[0]]

        for i in range(1, 2 * self.marge_range-1):
            reg0 = open3d.pipelines.registration.registration_icp(
                pcd0_list[i], pcd0_list[0], 5.0, np.eye(4),
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=100000))
            pcd0_list[i].transform(reg0.transformation)
            pcd0_list_aligned.append(pcd0_list[i])
            # open3d.visualization.draw_geometries([pcd0_list_aligned[0], pcd0_list_aligned[i]])

            reg1 = open3d.pipelines.registration.registration_icp(
                pcd1_list[i], pcd1_list[0], 5.0, np.eye(4),
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=100000))
            pcd1_list[i].transform(reg1.transformation)
            pcd1_list_aligned.append(pcd1_list[i])
            # open3d.visualization.draw_geometries([pcd1_list_aligned[0], pcd1_list_aligned[i]])

        # visualizer = open3d.visualization.Visualizer()
        # visualizer.create_window()
        # for p in pcd0_list_aligned:
        #     visualizer.add_geometry(p)
        # visualizer.get_render_option().show_coordinate_frame = True

        # visualizer.run()
        # visualizer.destroy_window()

        pcd0_m = open3d.geometry.PointCloud()
        pcd1_m = open3d.geometry.PointCloud()
        for m in pcd0_list_aligned:
            pcd0_m = pcd0_m + m
        for m in pcd1_list_aligned:
            pcd1_m = pcd1_m + m

        pcd0_k = open3d.geometry.keypoint.compute_iss_keypoints(pcd0_m,
                                                        salient_radius=1,
                                                        non_max_radius=1,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)
        pcd1_k = open3d.geometry.keypoint.compute_iss_keypoints(pcd1_m,
                                                        salient_radius=1,
                                                        non_max_radius=1,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)
        print(pcd0_k, pcd1_k)

        if aligned is True:
            reg = open3d.pipelines.registration.registration_icp(
                pcd0_k, pcd1_k, 15.0, np.eye(4),
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-13, relative_rmse=1e-13, max_iteration=100000))
            pcd0_m.transform(reg.transformation)

            pcd0_m, _ = pcd0_m.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            pcd1_m, _ = pcd1_m.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

            reg = open3d.pipelines.registration.registration_icp(
                pcd0_m, pcd1_m, 1.0, np.eye(4),
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-15, relative_rmse=1e-15, max_iteration=100000))
            pcd0_m.transform(reg.transformation)

            pcd0_m = pcd0_m.voxel_down_sample(voxel_size=downsample)
            pcd1_m = pcd1_m.voxel_down_sample(voxel_size=downsample)

            visualizer = open3d.visualization.Visualizer()
            visualizer.create_window()
            pcd0_m.paint_uniform_color([1, 0, 0])
            pcd1_m.paint_uniform_color([0, 0, 1])
            visualizer.add_geometry(pcd0_m)
            visualizer.add_geometry(pcd1_m)
            visualizer.get_render_option().show_coordinate_frame = True

            visualizer.run()
            visualizer.destroy_window()

        return np.array(pcd0_m.points), np.array(pcd1_m.points)

    # load_all_ply 메서드: 이 메서드는 모든 .ply 파일을 로드하고, 이를 다운샘플링한 다음 메모리에 보관합니다. 
    # 메모리에 로드된 데이터를 나중에 재사용하기 위해 .pkl 파일로 저장합니다.
    def load_all_ply(self, downsample):
        pts_filename = join(self.savepath, f'kitti_{self.split}_{downsample:.3f}_points.pkl')
        if exists(pts_filename):
            with open(pts_filename, 'rb') as file:
                
                self.pts = pickle.load(file)
            print(f"Load pts file from {self.savepath}")
            return
        self.pts = {}
        
        for drive in self.scene_list:
            idx_range = range(self.marge_range, len(self.scene_to_ids[drive])-self.marge_range, self.scan_gap)
            for i in idx_range:
                idx0 = self.scene_to_ids[drive][i]
                idx1 = self.scene_to_ids[drive][i+self.scan_gap]
                self.poses_pair.append((idx0, idx1))
                idx0_range = self.scene_to_ids[drive][i-self.marge_range:i+self.marge_range-1]
                idx1_range = self.scene_to_ids[drive][i+self.scan_gap-self.marge_range:i+self.scan_gap+self.marge_range-1]
                print(f"Load pts file... scan: {idx0_range}, {idx1_range}")
                root0 = []
                root1 = []
                for idx in idx0_range:
                    root0.append(os.path.join(self.root, idx + f'.bin'))
                for idx in idx1_range:
                    root1.append(os.path.join(self.root, idx + f'.bin'))
                print(root0)
                pcd0, pcd1 = self.load_ply(root0, root1, drive, i-self.marge_range, i+self.scan_gap-self.marge_range, downsample=downsample, aligned=True)
                self.pts[idx0] = pcd0
                self.pts[idx1] = pcd1

        with open(pts_filename, 'wb') as file:
            pickle.dump(self.pts, file)

    # get_matching_indices 메서드: 이 메서드는 두 개의 포인트 클라우드 간에 일치하는 포인트 쌍을 찾는 데 사용됩니다. 
    # 두 포인트 클라우드와 일치 기준(여기서는 거리)을 제공하면 일치하는 포인트 쌍의 인덱스를 반환합니다.
    def get_matching_indices(self, anc_pts, pos_pts, search_voxel_size, K=None):
        match_inds = []
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        match = bf_matcher.match(anc_pts, pos_pts)
        for match_val in match:
            if match_val.distance < search_voxel_size:
                match_inds.append([match_val.queryIdx, match_val.trainIdx])
        return np.array(match_inds)

    # cal_overlap 메서드: 이 메서드는 포인트 클라우드 간의 겹침 비율을 계산하고, 겹침 비율이 일정 값 이상인 키포인트 쌍을 식별합니다. 
    # 겹침 정보 및 키포인트 쌍을 .pkl 파일로 저장합니다.
    def cal_overlap(self, downsample):
        overlap_filename = join(self.savepath, f'kitti_{self.split}_{downsample:.3f}_overlap.pkl')
        keypts_filename = join(self.savepath, f'kitti_{self.split}_{downsample:.3f}_keypts.pkl')
        if exists(overlap_filename) and exists(keypts_filename):
            with open(overlap_filename, 'rb') as file:
                self.overlap_ratio = pickle.load(file)
                print(f"Reload overlap info from {overlap_filename}")
            with open(keypts_filename, 'rb') as file:
                self.keypts_pairs = pickle.load(file)
                print(f"Reload keypts info from {keypts_filename}")
            import pdb
            pdb.set_trace()
            return
        t0 = time.time()

        # scene은 폴더이름(kitti는 seq), scene_ids는 각 파일의 이름
        ## 즉 여기 processing을 해야함. self.poses_pair
        scene_overlap = {}
        print(f"Begin processing scene")
        for i, (idx0, idx1) in enumerate(self.poses_pair):
            anc_pts = self.pts[idx0].astype(np.float32)
            pos_pts = self.pts[idx1].astype(np.float32)

            try:
                # 이걸 keypts_pairs.pks에 저장하는 것!
                # 일치하는 포인트 쌍의 인덱스를 반환
                matching_01 = self.get_matching_indices(anc_pts, pos_pts, self.downsample)
            except BaseException as e:
                print(f"Something wrong with get_matching_indices {e} for {idx0}, {idx1}")
                matching_01 = np.array([])
            if len(anc_pts) == 0:
                print("division by zero")
                continue
            overlap_ratio = len(matching_01) / len(anc_pts)
            print(overlap_ratio)
            scene_overlap[f'{idx0}@{idx1}'] = overlap_ratio
            if overlap_ratio > 0.30:
                self.keypts_pairs[f'{idx0}@{idx1}'] = matching_01.astype(np.int32)
                self.overlap_ratio[f'{idx0}@{idx1}'] = overlap_ratio
                print(f'\t {idx0}, {idx1} overlap ratio: {overlap_ratio}')
            print('processing : {:.1f}%'.format(100 * i / len(self.poses_pair)))
        print('Finish, Done in {:.1f}s'.format(time.time() - t0))

        with open(overlap_filename, 'wb') as file:
            pickle.dump(self.overlap_ratio, file)
        with open(keypts_filename, 'wb') as file:
            pickle.dump(self.keypts_pairs, file)

if __name__ == '__main__':
    Kitti_cal_overlap(root='/media/vision/Seagate/DataSets/kitti/dataset/sequences',
                    pose_root='/media/vision/Seagate/DataSets/kitti/dataset/poses',
                    savepath='../data/kitti2',
                    split='train',
                    downsample=0.30
                    )
