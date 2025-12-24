import io, os
import json, h5py
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

abs_datasets_dir = '~/yanfeng/data/robotics/maniskill2_uniview'
abs_datasets_dir = os.path.join(abs_datasets_dir, "v0")

def find_files_with_suffix(root_path, suffix):
    import glob
    search_pattern = os.path.join(root_path, '**', f'*{suffix}') # 构建搜索模式
    matching_files = glob.glob(search_pattern, recursive=True) # 在指定路径下递归搜索所有符合后缀的文件
    absolute_paths = [os.path.abspath(file) for file in matching_files] # 获取并返回这些文件的绝对路径
    return absolute_paths
h5_files = find_files_with_suffix(abs_datasets_dir, 'rgbd.pd_ee_delta_pose.h5')

if 1:
    def process_file(file_idx, frames):

        from scipy.spatial.transform import Rotation as R
        def base_pose_to_matrix(base_pose):
            position = base_pose[:3] # 提取位置和四元数
            orientation = base_pose[3:]
            rotation = R.from_quat(orientation).as_matrix() # 将四元数转换为旋转矩阵
            transform_matrix = np.eye(4) # 创建4x4的同质变换矩阵
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = position
            return transform_matrix

        action = frames["actions"][file_idx]

        calib = {'rgb_static': {'extrinsic_matrix': frames["obs"]["camera_param"]["base_camera"]['extrinsic_cv'][file_idx],          
                            'intrinsic_matrix': frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][file_idx],
                            'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                'rgb_gripper': {'extrinsic_matrix': frames["obs"]["camera_param"]["hand_camera"]['extrinsic_cv'][file_idx],
                            'intrinsic_matrix': frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][file_idx],
                            'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}
        base_pose  = base_pose_to_matrix(frames['obs']['agent']['base_pose'][file_idx])

        gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_pose)
        loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))

        if 0:
            rgb_static = frames["obs"]["image"]["base_camera"]["rgb"][file_idx]
            rgb_gripper = frames["obs"]["image"]["hand_camera"]["rgb"][file_idx]
            depth_static = np.squeeze(frames["obs"]["image"]["base_camera"]["depth"][file_idx])
            depth_gripper = np.squeeze(frames["obs"]["image"]["hand_camera"]["depth"][file_idx])
            calib = {'rgb_static': {'extrinsic_matrix': frames["obs"]["camera_param"]["base_camera"]['extrinsic_cv'][file_idx],          
                                    'intrinsic_matrix': frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][file_idx],
                                    'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                    'rgb_gripper': {'extrinsic_matrix': frames["obs"]["camera_param"]["hand_camera"]['extrinsic_cv'][file_idx],
                                    'intrinsic_matrix': frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][file_idx],
                                    'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}
            def calculate_fov(intrinsic_matrix, image_width):
                fx = intrinsic_matrix[0, 0]
                fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
                return fov
            cam_config = {'static':{'height': rgb_static.shape[0],
                                    'width': rgb_static.shape[1],
                                    'fov': calculate_fov(frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][file_idx], rgb_static.shape[0])}, #90
                        'gripper':{'height': rgb_gripper.shape[0],
                                    'width': rgb_gripper.shape[1],
                                    'fov': calculate_fov(frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][file_idx], rgb_gripper.shape[0])}}
            # joints = frames["obs"]["agent"]["base_pose"][file_idx]

            ep = {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper,
                "depth_static": depth_static / 1000,
                "depth_gripper": depth_gripper/ 1000, # 单位mm
                "rel_actions": action,
                "robot_obs": np.zeros(20),
                "scene_obs": np.zeros(20),
                "calib": calib,
                "cam_config": cam_config,
                "base_pose": base_pose_to_matrix(frames['obs']['agent']['base_pose'][file_idx])
            }

            rgb = {}
            calib = ep['calib']
            cam_config = ep['cam_config']
            depth_static = ep['depth_static']   
            depth_gripper = ep['depth_gripper']
            static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   #注意这里没有求逆
            gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
            # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
            static_extrinsic_matrix = np.dot(static_extrinsic_matrix, ep["base_pose"])
            gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, ep["base_pose"])

            static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width'], cam_config['static']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width'], cam_config['gripper']['fov'])
            static_pcd = deproject(
                static_cam, depth_static,  #注意这里没有翻转
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = deproject(
                gripper_cam, depth_gripper, #注意这里没有翻转
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            cloud = np.concatenate([static_pcd,gripper_pcd],axis=0)

            rgb['rgb_static'] = Image.fromarray(ep['rgb_static'])
            rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper'])
            rgb['rgb_static'] = np.array(rgb['rgb_static'])   #注意这里没有翻转
            rgb['rgb_gripper'] = np.array(rgb['rgb_gripper']) #注意这里没有翻转
            static_rgb = np.reshape(
                rgb['rgb_static'], ( rgb['rgb_static'].shape[0] * rgb['rgb_static'].shape[1], 3)
            )
            gripper_rgb = np.reshape(
                rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0] * rgb['rgb_gripper'].shape[1], 3)
            )
            pcd_rgb = np.concatenate([static_rgb, gripper_rgb],axis=0)
            pcd_rgb = pcd_rgb/255
            if 1:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cloud)
                pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                o3d.io.write_point_cloud("tmp.pcd", pcd)

        return loc, action

    def process_task(h5_f): 
        json_path = h5_f.replace(".h5", ".json")
        json_data = json.load(open(json_path, 'rb'))
        episodes = json_data["episodes"]
        data = h5py.File(h5_f, "r")
        for episode_id in range(len(episodes)):
            frames = data[f"traj_{episode_id}"]

            files = list(range(episodes[episode_id]["elapsed_steps"]))
            
            results = [process_file(i, frames) for i in files]
            # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            #     results = [executor.submit(process_file, i, frames) for i in files]

            # for future in concurrent.futures.as_completed(results):
            #     gripper_loc,action= future.result()
            for gripper_loc, action in results:
                if len(action) == 7:
                    actions.append(action)
                else:
                    print(action)
                    print(h5_f)
                if len(gripper_loc) == 4:
                    gripper_locs.append(gripper_loc)
                else:
                    print(gripper_loc)
        data.close()
        
    gripper_locs = []
    actions = []
    for h5_f in h5_files:
        process_task(h5_f)

    gripper_locs = np.stack(gripper_locs)
    actions=np.stack(actions)
    print(gripper_locs.min(0), gripper_locs.max(0))
    print(actions.min(0), actions.max(0))
    np.save('data_folder/robocasa_gripper_limit.npy', gripper_locs)
    np.save('data_folder/robocasa_actions.npy', actions)
    for i in range(3):
        print(plt.hist(gripper_locs[:, i], bins=20, color='blue', edgecolor='black'))
    for i in range(7):
        print(plt.hist(actions[:, i], bins=20, color='blue', edgecolor='black'))
