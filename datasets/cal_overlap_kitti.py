import os
from os.path import exists, join
import pickle
import numpy as np
import open3d
import cv2
import time
import copy
import matplotlib.pyplot as plt

def odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    T_w_cam0 = np.vstack((T_w_cam0, [0.0, 0.0, 0.0, 1.0]))
    return T_w_cam0

def loadPoses(self):
    # load poses
    filePoses = os.path.join(self.kitti360Path, 'data_poses', self.sequence, 'poses.txt')
    poses = np.loadtxt(filePoses)
    frames = poses[:,0]
    poses = np.reshape(poses[:,1:],[-1,3,4])
    self.Tr_pose_world = {}
    self.frames = frames
    for frame, pose in zip(frames, poses): 
        pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
        self.Tr_pose_world[frame] = pose


class Kitti_cal_overlap(object):
    """
    Given point cloud fragments and corresponding pose in '{root}'.
        1. Save the aligned point cloud pts in '{savepath}/3DMatch_{downsample}_points_ver2.pkl'
        2. Calculate the overlap ratio and save in '{savepath}/3DMatch_{downsample}_overlap_ver2.pkl'
        3. Save the ids of anchor keypoints and positive keypoints in '{savepath}/3DMatch_{downsample}_keypts_ver2.pkl'
    """

    # __init__ 메서드: 이 메서드는 클래스의 초기화를 처리합니다. 데이터셋의 루트 디렉토리 (root), 결과를 저장할 경로 (savepath), 
    # 데이터셋 분할 (split), 다운샘플링 크기 (downsample)를 인수로 받습니다. 이 메서드에서는 필요한 초기화 작업을 수행하고 데이터셋을 불러오는 데 사용됩니다.
    def __init__(self, root, pose_root, savepath, split, downsample):
        # 초기 빈 그래프 생성
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([])  # 빈 라인 생성
        self.root = root
        self.pose_root = pose_root
        self.savepath = savepath
        os.makedirs(self.savepath, exist_ok=True)
        self.split = split
        self.downsample = downsample
        self.poses = {}
        self.MIN_DIST = 25.0
        self.Num_of_splits = 4
        self.crop_distance = 40
        self.poses_pair = []
        self.poses = {}
        self.pair_keys = []
        self.T = []

        # dict: from id to pts.
        self.pts = {}
        # dict: from id_id to overlap_ratio
        self.overlap_ratio = {}
        # dict: from id_id to anc_keypts id & pos_keypts id
        self.keypts_pairs = {}

        # seq 목록 불러오기
        # if self.split == 'train': self.scene_list = ["00"] # 테스트용
        if self.split == 'train': self.scene_list = ["00","01","02","03","04","05","06"]
        elif self.split == 'val': self.scene_list = ["07","08"] 
        else: self.scene_list = ["09","10"] 

        self.ids_list = []
        self.ids_list_count = {}
        self.scene_to_ids = {} # 결국 이거 얻기 위함
        
        for scene in self.scene_list:
            self.ids_list_count[scene] = len(self.ids_list)
            # poses 불러오기
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
            
            self.make_poses_pair(scene)

            print(f"Seq {scene}: num bin: {len(ids)}")
        print(f"Total {len(self.scene_list)} scenes, {len(self.ids_list)} point cloud fragments.")
        print(f"Total {len(self.poses_pair)} pair, {self.MIN_DIST}m apart each point cloud fragments.")
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
        
    def make_poses_pair(self, drive):
        Ts = self.poses[drive][:, :3, 3]
        Ts_expanded = Ts[:, np.newaxis, :] 
        pdist = np.linalg.norm(Ts_expanded - Ts, axis=-1) 
        valid_pairs = pdist > self.MIN_DIST

        inames = list(range(len(valid_pairs)))
        curr_time = inames[0]
        mid_ids_list = []
        while curr_time in inames:
          next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 1000])[0]
          if len(next_time) == 0:
            curr_time += 1
          else:
            # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
            next_time = next_time[0] + curr_time - 1

          if next_time in inames:
            for a in range(curr_time + 1, next_time, 1):
                mid_ids_list.append(self.ids_list[a + self.ids_list_count[drive]])
            self.poses_pair.append((self.ids_list[curr_time + self.ids_list_count[drive]], self.ids_list[next_time + self.ids_list_count[drive]], mid_ids_list))
            curr_time = (next_time + 1) - (5*(next_time - curr_time)//7)
            mid_ids_list = []
            
    # load_ply 메서드: 이 메서드는 주어진 디렉토리에서 .ply 포맷의 3D 포인트 클라우드 데이터를 로드합니다. 
    # 다운샘플링을 수행하고, 포즈 정보를 사용하여 포인트 클라우드를 정렬합니다.
    def load_ply(self, fname0, fname1, fname_mid, downsample, aligned=True): #join(data_dir, f'{ind}.bin')
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        distance_threshold = 4.0
        mask = np.linalg.norm(xyz0[:, :3], axis=1) > distance_threshold
        xyz0 = xyz0[mask]
        mask = np.linalg.norm(xyz1[:, :3], axis=1) > distance_threshold
        xyz1 = xyz1[mask]

        pcd0 = open3d.geometry.PointCloud()
        pcd3 = open3d.geometry.PointCloud()
        pcd0.points = open3d.utility.Vector3dVector(xyz0)
        pcd3.points = open3d.utility.Vector3dVector(xyz1)
        pcd0 = pcd0.voxel_down_sample(voxel_size=downsample)
        pcd3 = pcd3.voxel_down_sample(voxel_size=downsample)

        pcd_mids = []
        for a in fname_mid:
            xyzr_mid = np.fromfile(a, dtype=np.float32).reshape(-1, 4)
            xyz_mid = xyzr_mid[:, :3]
            mask = np.linalg.norm(xyz_mid[:, :3], axis=1) > distance_threshold
            xyz_mid = xyz_mid[mask]
            pcd_mid = open3d.geometry.PointCloud()
            pcd_mid.points = open3d.utility.Vector3dVector(xyz_mid)
            pcd_mid = pcd_mid.voxel_down_sample(voxel_size=downsample)
            pcd_mids.append(pcd_mid)
        pcd_mids.append(pcd3)

        mid_idx = len(pcd_mids) // self.Num_of_splits  # 리스트의 중간 인덱스 계산 -> 하나만 쓰려면 mid_idx = 1
        # mid_idx = 1

        # 중간을 기준으로 리스트를 나누기
        first_half = pcd_mids[:mid_idx]
        second_half = pcd_mids[mid_idx:len(pcd_mids) - mid_idx]
        last_half = pcd_mids[len(pcd_mids) - mid_idx:]
        
        pcd1 = open3d.geometry.PointCloud()
        pcd1 = pcd_mids[mid_idx]
        pcd2 = open3d.geometry.PointCloud()
        pcd2 = pcd_mids[(self.Num_of_splits-1)*mid_idx]

        T_accumulated = np.eye(4)

        if aligned is True:
            for pcd_mid in first_half:
                reg = open3d.pipelines.registration.registration_icp(
                    pcd0, pcd_mid, 2.0, np.eye(4),
                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=100000))
                pcd0.transform(reg.transformation)
                pcd0 = pcd0 + pcd_mid
                pcd0 = pcd0.voxel_down_sample(voxel_size=downsample)
                T_accumulated = np.dot(T_accumulated, reg.transformation)

            for pcd_mid in second_half:
                reg = open3d.pipelines.registration.registration_icp(
                    pcd1, pcd_mid, 2.0, np.eye(4),
                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=100000))
                # 잘 됬는지 확인
                pcd0.transform(reg.transformation)
                pcd1.transform(reg.transformation)
                pcd1 = pcd1 + pcd_mid
                pcd1 = pcd1.voxel_down_sample(voxel_size=downsample)
                T_accumulated = np.dot(T_accumulated, reg.transformation)
            
            for pcd_mid in last_half:
                reg = open3d.pipelines.registration.registration_icp(
                    pcd2, pcd_mid, 2.0, np.eye(4),
                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=100000))
                pcd0.transform(reg.transformation)
                pcd1.transform(reg.transformation)
                pcd2.transform(reg.transformation)
                pcd2 = pcd2 + pcd_mid
                pcd2 = pcd2.voxel_down_sample(voxel_size=downsample)
                T_accumulated = np.dot(T_accumulated, reg.transformation)

            pcd0, _ = pcd0.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)
            pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

            pcd0 = pcd0.voxel_down_sample(voxel_size=downsample)
            pcd2 = pcd2.voxel_down_sample(voxel_size=downsample)
            
            reg = open3d.pipelines.registration.registration_icp(
                    pcd0, pcd2, 2.0, np.eye(4),
                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=100000))
            pcd0.transform(reg.transformation)
            T_accumulated = np.dot(T_accumulated, reg.transformation)

            # 무게 중심을 기준으로 일정 거리 안쪽의 점을 크롭
            centroid = pcd0.get_center()
            crop_box = open3d.geometry.AxisAlignedBoundingBox(
                min_bound=[centroid[0] - self.crop_distance, centroid[1] - self.crop_distance, centroid[2] - self.crop_distance],
                max_bound=[centroid[0] + self.crop_distance, centroid[1] + self.crop_distance, centroid[2] + self.crop_distance]
            )
            pcd0 = pcd0.crop(crop_box)

            centroid = pcd2.get_center()
            crop_box = open3d.geometry.AxisAlignedBoundingBox(
                min_bound=[centroid[0] - self.crop_distance, centroid[1] - self.crop_distance, centroid[2] - self.crop_distance],
                max_bound=[centroid[0] + self.crop_distance, centroid[1] + self.crop_distance, centroid[2] + self.crop_distance]
            )
            pcd2 = pcd2.crop(crop_box)
            
            # visualizer = open3d.visualization.Visualizer()
            # visualizer.create_window()
            # pcd0.paint_uniform_color([1, 0, 0])
            # pcd2.paint_uniform_color([0, 0, 1])
            # visualizer.add_geometry(pcd0)
            # visualizer.add_geometry(pcd2)
            # visualizer.get_render_option().show_coordinate_frame = True
            # visualizer.run()
            # visualizer.destroy_window()

            if np.linalg.norm(T_accumulated[:3, 3]) < self.MIN_DIST *0.8:
                print("icp process failed")
                return np.array(pcd0.points), np.array(pcd1.points), False
            self.T.append(T_accumulated)
            ## 다시 원래 좌표계
            # restored_coordinates = np.linalg.inv(T_accumulated)
            # pcd0.transform(restored_coordinates)
        return np.array(pcd0.points), np.array(pcd2.points), True

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

        for i, (idx0, idx1, idxlist) in enumerate(self.poses_pair):
            print(f"Load pts file... scan: {idx0}, {idx1}")
            if idx0 in self.pts:
                print(f"{idx0} aready in pts")
                continue
            if idx1 in self.pts:
                print(f"{idx1} aready in pts")
                continue
            root0 = os.path.join(self.root, idx0)
            root1 = os.path.join(self.root, idx1)
            root_list = []
            for a in idxlist:
                a = a + f'.bin'
                root_list.append(os.path.join(self.root, a))
            pcd0, pcd1, flag= self.load_ply((root0 + f'.bin'), (root1 + f'.bin'), root_list, downsample=downsample, aligned=True)
            if flag:
                self.pts[idx0] = pcd0
                self.pts[idx1] = pcd1
                self.pair_keys.append([idx0, idx1])
                if self.split =='test':
                    self.pts[idx1] = pcd1
                    self.pts[idx0] = pcd0
                    self.pair_keys.append([idx1, idx0])
            else:
                print("this pair not align!")
            print('processing : {:.1f}%'.format(100 * i / len(self.poses_pair)))

        with open(pts_filename, 'wb') as file:
            pickle.dump(self.pts, file)

    # get_matching_indices 메서드: 이 메서드는 두 개의 포인트 클라우드 간에 일치하는 포인트 쌍을 찾는 데 사용됩니다. 
    # 두 포인트 클라우드와 일치 기준(여기서는 거리)을 제공하면 일치하는 포인트 쌍의 인덱스를 반환합니다.
    def get_matching_indices(self, anc_pts, pos_pts, T_idx ,search_voxel_size, K=None):
        match_inds = []
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        match = bf_matcher.match(anc_pts, pos_pts)
        for match_val in match:
            if match_val.distance < search_voxel_size * 1.5:
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
        ## 즉 여기 processing을 해야함.
        scene_overlap = {}
        print(f"Begin processing scene")
        for i, key_idx in enumerate(self.pair_keys):
            anc_pts = self.pts[key_idx[0]].astype(np.float32)
            pos_pts = self.pts[key_idx[1]].astype(np.float32)
            
            try:
                pass
                # 이걸 keypts_pairs.pks에 저장하는 것!
                # 일치하는 포인트 쌍의 인덱스를 반환
                matching_01 = self.get_matching_indices(anc_pts, pos_pts, i, self.downsample)
            except BaseException as e:
                print(f"Something wrong with get_matching_indices {e} for {key_idx[0]}, {key_idx[1]}")
                matching_01 = np.array([])
            if len(anc_pts) == 0:
                print("division by zero")
                continue
            overlap_ratio = len(matching_01) / len(anc_pts)
            # print(overlap_ratio)
            scene_overlap[f'{key_idx[0]}@{key_idx[1]}'] = overlap_ratio
            if overlap_ratio > 0.25:
                self.keypts_pairs[f'{key_idx[0]}@{key_idx[1]}'] = matching_01.astype(np.int32)
                self.overlap_ratio[f'{key_idx[0]}@{key_idx[1]}'] = overlap_ratio
                print(f'\t {key_idx[0]}, {key_idx[1]} overlap ratio: {overlap_ratio}')
            print('processing : {:.1f}%'.format(100 * i / len(self.pair_keys)))
        print('Finish, Done in {:.1f}s'.format(time.time() - t0))

        with open(overlap_filename, 'wb') as file:
            pickle.dump(self.overlap_ratio, file)
        with open(keypts_filename, 'wb') as file:
            pickle.dump(self.keypts_pairs, file)


if __name__ == '__main__':
    for t in ["train", "val", "test"]: # ["train", "val", "test"]: ['test']
        Kitti_cal_overlap(root='/media/vision/Seagate/DataSets/kitti/dataset/sequences',
                        pose_root='/media/vision/Seagate/DataSets/kitti/dataset/poses',
                        savepath='../data/kitti/ds03_ver3',
                        split=t,
                        downsample=0.3
                        )
