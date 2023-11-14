import os
from os.path import exists, join
import pickle
import numpy as np
import open3d
import cv2
import time


class ThreeDMatch(object):
    """
    Given point cloud fragments and corresponding pose in '{root}'.
        1. Save the aligned point cloud pts in '{savepath}/3DMatch_{downsample}_points.pkl'
        2. Calculate the overlap ratio and save in '{savepath}/3DMatch_{downsample}_overlap.pkl'
        3. Save the ids of anchor keypoints and positive keypoints in '{savepath}/3DMatch_{downsample}_keypts.pkl'
    """

    # __init__ 메서드: 이 메서드는 클래스의 초기화를 처리합니다. 데이터셋의 루트 디렉토리 (root), 결과를 저장할 경로 (savepath), 
    # 데이터셋 분할 (split), 다운샘플링 크기 (downsample)를 인수로 받습니다. 이 메서드에서는 필요한 초기화 작업을 수행하고 데이터셋을 불러오는 데 사용됩니다.
    def __init__(self, root, savepath, split, downsample):
        self.root = root
        self.savepath = savepath
        self.split = split
        self.downsample = downsample

        # dict: from id to pts.
        self.pts = {}

        # dict: from id_id to overlap_ratio
        self.overlap_ratio = {}
        # dict: from id_id to anc_keypts id & pos_keypts id
        self.keypts_pairs = {}

        with open(os.path.join(root, f'scene_list_{split}.txt')) as f:
            scene_list = f.readlines()
        self.ids_list = []
        self.scene_to_ids = {}
        
        for scene in scene_list:
            scene = scene.replace("\n", "")
            # print(scene)
            self.scene_to_ids[scene] = []
            # print(self.scene_to_ids)

            for seq in sorted(os.listdir(os.path.join(self.root, scene))):
                # print(seq)
                if not seq.startswith('seq'):
                    continue
                scene_path = os.path.join(self.root, scene + f'/{seq}')
                
                ids = [scene + f"/{seq}/" + str(filename.split(".")[0]) for filename in os.listdir(scene_path) if filename.endswith('ply')]

                ids = sorted(ids, key=lambda x: int(x.split("/")[-1]))
                self.ids_list += ids
                self.scene_to_ids[scene] += ids
                print(ids)
                print(f"Scene {scene}, seq {seq}: num ply: {len(ids)}")
        print(f"Total {len(scene_list)} scenes, {len(self.ids_list)} point cloud fragments.")
        self.idpair_list = []
        print("----------------------------")
        # print(self.scene_to_ids) ## >>  'analysis-by-synthesis-office2-5b': ['analysis-by-synthesis-office2-5b/
        # print(downsample)
        self.load_all_ply(downsample)
        self.cal_overlap(downsample)

    # load_ply 메서드: 이 메서드는 주어진 디렉토리에서 .ply 포맷의 3D 포인트 클라우드 데이터를 로드합니다. 
    # 다운샘플링을 수행하고, 포즈 정보를 사용하여 포인트 클라우드를 정렬합니다.
    def load_ply(self, data_dir, ind, downsample, aligned=True):
        print(downsample)
        pcd = open3d.io.read_point_cloud(join(data_dir, f'{ind}.ply'))
        pcd = pcd.voxel_down_sample(voxel_size=downsample)
        # if aligned is True:
        #     matrix = np.load(join(data_dir, f'{ind}.pose.npy'))
        #     pcd.transform(matrix)
        return pcd

    # load_all_ply 메서드: 이 메서드는 모든 .ply 파일을 로드하고, 이를 다운샘플링한 다음 메모리에 보관합니다. 
    # 메모리에 로드된 데이터를 나중에 재사용하기 위해 .pkl 파일로 저장합니다.
    def load_all_ply(self, downsample):
        pts_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_points.pkl')
        if exists(pts_filename):
            with open(pts_filename, 'rb') as file:
                self.pts = pickle.load(file)
            print(f"Load pts file from {self.savepath}")
            return
        self.pts = {}
        for i, anc_id in enumerate(self.ids_list):
            anc_pcd = self.load_ply(self.root, anc_id, downsample=downsample, aligned=True)
            points = np.array(anc_pcd.points)
            print(len(points))
            self.pts[anc_id] = points
            print('processing ply: {:.1f}%'.format(100 * i / len(self.ids_list)))

        print(type(self.pts[self.ids_list[0]]))
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
        overlap_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_overlap.pkl')
        keypts_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_keypts.pkl')
        # 두 파일 이미 있으면 reload
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

        for scene, scene_ids in self.scene_to_ids.items():
            scene_overlap = {}
            print(f"Begin processing scene {scene}")
            for i in range(0, len(scene_ids)):
                anc_id = scene_ids[i]
                for j in range(i + 1, len(scene_ids)):
                    pos_id = scene_ids[j]
                    anc_pts = self.pts[anc_id].astype(np.float32)
                    pos_pts = self.pts[pos_id].astype(np.float32)

                    try:
                        matching_01 = self.get_matching_indices(anc_pts, pos_pts, self.downsample)
                    except BaseException as e:
                        print(f"Something wrong with get_matching_indices {e} for {anc_id}, {pos_id}")
                        matching_01 = np.array([])
                    if len(anc_pts) == 0:
                        print("division by zero")
                        continue
                    overlap_ratio = len(matching_01) / len(anc_pts)
                    # matching_10 = self.get_matching_indices(pos_pts, anc_pts, self.downsample)
                    # overlap_ratio = max(len(matching_01) / len(anc_pts), len(matching_10) / len(pos_pts))

                    scene_overlap[f'{anc_id}@{pos_id}'] = overlap_ratio
                    if overlap_ratio > 0.30:
                        self.keypts_pairs[f'{anc_id}@{pos_id}'] = matching_01.astype(np.int32)
                        self.overlap_ratio[f'{anc_id}@{pos_id}'] = overlap_ratio
                        print(f'\t {anc_id}, {pos_id} overlap ratio: {overlap_ratio}')
                print('processing {:s} ply: {:.1f}%'.format(scene, 100 * i / len(scene_ids)))
            print('Finish {:s}, Done in {:.1f}s'.format(scene, time.time() - t0))

        with open(overlap_filename, 'wb') as file:
            pickle.dump(self.overlap_ratio, file)
        with open(keypts_filename, 'wb') as file:
            pickle.dump(self.keypts_pairs, file)


if __name__ == '__main__':
    ThreeDMatch(root='/media/vision/Seagate/DataSets/3DMatch',
                savepath='../data/3DMatch',
                split='train',
                downsample=0.025
                )
