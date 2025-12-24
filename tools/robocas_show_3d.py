import json
import io, os
import re,h5py
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import hydra, pybullet
from scipy.spatial.transform import Rotation as scipyR 

abs_datasets_dir = Path('~/zhengliming/isaac-dataset/isaac_pick_place-rand-rbox')
with open(os.path.join(abs_datasets_dir, 'data_info.json'), 'r') as f:
    info = json.load(f)

if 1:
    def process_file(file_idx, episodes_dir, episode_info):
        try:
            def angle_between_angles(a, b):
                diff = b - a
                return (diff + np.pi) % (2 * np.pi) - np.pi
            def to_relative_action(actions, robot_obs, max_pos=1, max_orn=1):
                rel_pos = actions[:3] - robot_obs[:3]
                rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
                rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
                rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
                gripper = actions[-1:]
                return np.concatenate([rel_pos, rel_orn, gripper])
            r_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx, [4,5,6,3]]))
            a_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx+1, [4,5,6,3]]))
            if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8: # 大于0.8就发生了万向节死锁
                sr_euler = np.array(scipyR.from_quat(frames[file_idx].gripper_pose[3:7]).as_euler('xyz', degrees=False))
                sa_euler = np.array(scipyR.from_quat(frames[file_idx+1].gripper_pose[3:7]).as_euler('xyz', degrees=False))
                if np.abs(angle_between_angles(a_euler, r_euler) - angle_between_angles(sa_euler, sr_euler)).max()<0.01: # 并没有发生万向节死锁
                    pass
                elif np.abs(angle_between_angles(sa_euler, sr_euler)).max() < 0.8: # 大于0.8就发生了万向节死锁
                    r_euler = sr_euler
                    a_euler = sa_euler
                else:
                    if np.abs(angle_between_angles(sa_euler, r_euler)).max() < 0.8:
                        a_euler = sa_euler
                    elif np.abs(angle_between_angles(a_euler, sr_euler)).max() < 0.8:
                        r_euler = sr_euler
                    else:
                        pass
            robot_obs_euler = np.concatenate([episode_info['ee_pose_root'][file_idx,:3], r_euler, [episode_info['gripper_target'][file_idx]]]) # YF: gripper_open=0/0.08, 注意要求的是-1和1
            actions_euler = np.concatenate([episode_info['ee_pose_root'][file_idx+1,:3], a_euler, [episode_info['gripper_target'][file_idx+1]]])
            action = to_relative_action(actions_euler, robot_obs_euler[:6])
                
            state = episode_info['ee_pose_root'][file_idx]
            arm_pose_mat = np.eye(4)
            arm_pose_mat[:3, 3] = state[:3]
            arm_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(state[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz', state[3:6], False).as_matrix()

            static_camera_pose, gripper_camera_pose = episode_info['base_camera'].item()['pose'], episode_info['gripper_camera'].item()['pose']
            static_pose_mat = np.eye(4)
            static_pose_mat[:3, 3] = static_camera_pose[:3] #   - np.array([0.7, 0, 0.6])
            static_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(static_camera_pose[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz',  R.from_quat(static_camera_pose[3:][[1, 2, 3, 0]]).as_euler('xyz', degrees=False), False).as_matrix()
            gripper_pose_mat = np.eye(4)
            gripper_pose_mat[:3, 3] = gripper_camera_pose[:3]
            gripper_pose_mat[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(gripper_camera_pose[[4,5,6,3]])).reshape(3,3) # R.from_euler('xyz',  R.from_quat(gripper_camera_pose[3:][[1, 2, 3, 0]]).as_euler('xyz', degrees=False), False).as_matrix()

            calib = {'rgb_static': {'extrinsic_matrix': static_pose_mat,
                                    'intrinsic_matrix': episode_info['base_camera'].item()['intrinsics'],
                                    'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                    'rgb_gripper': {'extrinsic_matrix': np.matmul(arm_pose_mat, gripper_pose_mat),
                                    'intrinsic_matrix': episode_info['gripper_camera'].item()['intrinsics'],
                                    'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}

            gripper_extrinsic_matrix = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            gripper_loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))
             
            # if gripper_loc[2] < 0:
            #     print(gripper_loc)

            if 0:
                        
                rgb_static = np.array(Image.open(os.path.join(episodes_dir, "base_camera/rgb/%04d.png"%file_idx)))
                rgb_gripper = np.array(Image.open(os.path.join(episodes_dir, "gripper_camera/rgb/%04d.png"%file_idx)))

                cam_config = {'static':{'height': rgb_static.shape[0],
                                        'width': rgb_static.shape[1],
                                        'fov': np.degrees(episode_info['base_camera'].item()['fovy'])}, #80
                            'gripper':{'height': rgb_gripper.shape[0],
                                        'width': rgb_gripper.shape[1],
                                        'fov': np.degrees(episode_info['gripper_camera'].item()['fovy'])}}

                ep = {
                    "rgb_static": rgb_static,
                    "rgb_gripper": rgb_gripper,
                    "rel_actions": action,
                    "robot_obs": np.zeros(20),
                    "scene_obs": np.zeros(20),
                    "depth_static": np.array(Image.open(os.path.join(episodes_dir, "base_camera/depth/%04d.png"%file_idx)))/1000,
                    "depth_gripper": np.array(Image.open(os.path.join(episodes_dir, "gripper_camera/depth/%04d.png"%file_idx)))/1000,
                    "calib":calib,
                    "cam_config": cam_config,
                }

                rgb = {}
                calib = ep['calib']
                cam_config = ep['cam_config']
                depth_static = ep['depth_static'][:, 80:-80]
                depth_gripper = ep['depth_gripper'][:, 80:-80]
                static_extrinsic_matrix = np.linalg.inv(calib['rgb_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])   #注意这里没有求逆
                gripper_extrinsic_matrix = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
                static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width']-80*2, cam_config['static']['fov']) # 此处因为抠图
                gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width']-80*2, cam_config['gripper']['fov'])
                static_pcd = deproject(
                    static_cam, depth_static,  #注意这里没有翻转
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                gripper_pcd = deproject(
                    gripper_cam, depth_gripper, #注意这里没有翻转
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                cloud = np.concatenate([static_pcd,gripper_pcd],axis=0)

                rgb['rgb_static'] = Image.fromarray(ep['rgb_static'][:, 80:-80])
                rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper'][:, 80:-80])
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
                    pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
                    o3d.io.write_point_cloud("tmp.pcd", pcd)
                    pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
                    o3d.io.write_point_cloud("tmp1.pcd", pcd)

            return action, gripper_loc
        except:
            return None, None

    gripper_locs = []
    actions = []

    episode_idx = info['train' + '_eps']
    for idx in episode_idx:
        episode_length = info['ep_length'][str(idx)]
        files = list(range(episode_length))
        episodes_dir = os.path.join(abs_datasets_dir, "episode_%07d" % idx)
        episode_info = np.load(os.path.join(episodes_dir, 'episode_info.npz'),allow_pickle=True)
        results = [process_file(i, episodes_dir, episode_info) for i in files]
        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #     results = [executor.submit(process_file, i, data_path) for i in files]
        # for future in concurrent.futures.as_completed(results):
        #     action, gripper_loc = future.result()
        for action, gripper_loc in results:
            if action is None: continue
            actions.append(action)
            gripper_locs.append(gripper_loc)

    gripper_locs=np.stack(gripper_locs)
    actions=np.stack(actions)
    print(gripper_locs.min(0), gripper_locs.max(0))
    print(actions.min(0), actions.max(0))
    np.save('data_folder/robocas_gripper_limit.py', gripper_locs)
    np.save('data_folder/robocas_actions.py', actions)
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))
    for i in range(7):
        print(plt.hist(actions[:,i], bins=20, color='blue', edgecolor='black'))


