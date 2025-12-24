import json
import io, os
import re,h5py
import numpy as np
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

abs_datasets_dir = '~/yanfeng/data/robotics/robomimic_uniview'
_task_list = os.listdir(abs_datasets_dir)
_task_list = ["lift", "can", "square"]# , "tool_hang"
if 1:

    def process_task(task, abs_datasets_dir):
        dataset_file = os.path.join(abs_datasets_dir, task, "ph", "depth.hdf5")
        data = h5py.File(dataset_file, "r")
        episodes = data["data"]             # 记录每一条轨迹的信息
        for i in range(len(episodes)):             # 构建episode_lookup
            episode_length = len(episodes["demo_{0}".format(i)]["actions"])
            frames = episodes["demo_{0}".format(i)]
            
            def process_file(file_idx, frames):
                action = frames["actions"][file_idx] # -1为开
                camera_info = json.loads(frames.attrs['camera_info_realtime'])
                extrinsic_matrix = camera_info[f"{file_idx}"]["robot0_eye_in_hand"]['fix_extrinsics']
                ep = {
                    "calib_gripper": {"extrinsic_matrix": extrinsic_matrix},
                }
                gripper_extrinsic_matrix = np.linalg.inv(ep['calib_gripper']['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
                gripper_loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))

                # if gripper_loc[2]< 0.7:
                #     print(gripper_loc)

                if 0:
                    rgb_static = frames["obs"]['agentview_image'][file_idx] # 256 256 3
                    rgb_gripper = frames["obs"]['robot0_eye_in_hand_image'][file_idx] # 256 256 3
                    depth_static = np.squeeze(frames["obs"]['agentview_depth'][file_idx])
                    depth_gripper = np.squeeze(frames["obs"]['robot0_eye_in_hand_depth'][file_idx])

                    camera_info = json.loads(frames.attrs['camera_info_realtime'])[f"{file_idx}"]

                    calib = {}
                    cam_config = {}
                    calib['rgb_static'] = {} 
                    calib['rgb_static']['extrinsic_matrix'] = np.linalg.inv(np.array(camera_info["agentview"]['extrinsics'])) # 4 4 
                    calib['rgb_static']['intrinsic_matrix'] = np.array(camera_info["agentview"]["intrinsics"]) # 3 3
                    calib['rgb_static']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8
                    calib['rgb_gripper'] = {}
                    calib['rgb_gripper']['extrinsic_matrix'] = np.linalg.inv(np.array(camera_info['robot0_eye_in_hand']['fix_extrinsics'])) # 4 4 
                    calib['rgb_gripper']['intrinsic_matrix'] = np.array(camera_info['robot0_eye_in_hand']["intrinsics"]) # 3 3
                    calib['rgb_gripper']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8

                    def calculate_fov(intrinsic_matrix, image_width):
                        fx = intrinsic_matrix[0, 0]
                        fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
                        return fov
                    cam_config = {}
                    cam_config['static'] = {}
                    cam_config['static']['height'] = rgb_static.shape[0]
                    cam_config['static']['width'] = rgb_static.shape[1]
                    cam_config['static']['fov'] = calculate_fov(calib['rgb_static']['intrinsic_matrix'], rgb_static.shape[0])
                    cam_config['gripper'] = {}
                    cam_config['gripper']['height'] = rgb_gripper.shape[0]
                    cam_config['gripper']['width'] = rgb_gripper.shape[1]
                    cam_config['gripper']['fov'] = calculate_fov(calib['rgb_gripper']['intrinsic_matrix'], rgb_gripper.shape[0])

                    joints = frames["obs"]["robot0_joint_pos"][file_idx]
                    ep= {
                        "rgb_static": rgb_static,
                        "rgb_gripper": rgb_gripper,
                        "depth_static": depth_static,
                        "depth_gripper": depth_gripper, # 单位m
                        # "rel_actions": action,
                        "robot_obs": joints,
                        "scene_obs": joints,
                        "calib": calib,
                        "cam_config": cam_config,
                    }

                    rgb = {}
                    calib = ep['calib']
                    cam_config = ep['cam_config']
                    depth_static = ep['depth_static']   
                    depth_gripper = ep['depth_gripper']
                    static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   #注意这里没有求逆
                    gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆

                    T_translate = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.4], # [-0.17879518 -0.40093808  0.90953124  1.        ] [0.33830822 0.33967275 1.29464515 1.        ]
                        [0, 0, 0, 1]
                    ])
                    static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
                    gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
                    R = np.array([
                        [0, 1, 0, 0],
                        [-1,  0, 0, 0],
                        [0,  0, 1, 0],
                        [0,  0, 0, 1],
                    ])
                    static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
                    gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
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
                    static_rgb =  np.reshape(
                        rgb['rgb_static'], ( rgb['rgb_static'].shape[0] * rgb['rgb_static'].shape[1], 3)
                    )
                    gripper_rgb =  np.reshape(
                        rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0] * rgb['rgb_gripper'].shape[1], 3)
                    )
                    pcd_rgb = np.concatenate([static_rgb, gripper_rgb],axis=0)
                    pcd_rgb = pcd_rgb/255
                    if 1:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(cloud)
                        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                        o3d.io.write_point_cloud("tmp.pcd", pcd)

                return action, gripper_loc


            files = list(range(0, episode_length))
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = [executor.submit(process_file, i, frames) for i in files]

            for future in concurrent.futures.as_completed(results):
                action, gripper_loc = future.result()
                actions.append(action)
                gripper_locs.append(gripper_loc)
    
    gripper_locs = []
    actions = []
    for task in _task_list:
        process_task(task, abs_datasets_dir)

    gripper_locs=np.stack(gripper_locs)
    print(gripper_locs.min(0), gripper_locs.max(0))
    np.save('data_folder/robomimic_gripper_limit.py', gripper_locs)
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))

    actions=np.stack(actions)
    print(actions.min(0), actions.max(0))
    # [-1.         -1.         -1.         -0.55634028 -1.         -1. -1.        ] [1.         1.         1.         0.72973686 0.45003703 1. 1.        ]
    np.save('data_folder/robomimic_actions.py', gripper_locs)
    for i in range(7):
        print(plt.hist(actions[:,i], bins=20, color='blue', edgecolor='black'))
