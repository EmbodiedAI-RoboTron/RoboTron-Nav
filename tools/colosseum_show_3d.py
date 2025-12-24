import io, os, cv2
import pickle
import numpy as np
from PIL import Image
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
import hydra, pybullet
from scipy.spatial.transform import Rotation as scipyR 

abs_datasets_dir = Path('~/huangyiyang02/data/robot-colsseum')
tasks = os.listdir(abs_datasets_dir)
gripper_locs, actions =[], []
for task_str in tasks:
    task_path = abs_datasets_dir / task_str / "variation0" / "episodes"
    for num_demo in os.listdir(task_path):
        task_name = f"{task_str}/variation0/episodes/{num_demo}"
        episodes_dir = abs_datasets_dir / task_name
        with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
            frames = pickle.load(f)

        for file_idx in range(len(frames)-1):
            def angle_between_angles(a, b):
                diff = b - a
                return  (diff + np.pi) % (2 * np.pi) - np.pi #  (diff + np.pi) % (2 * np.pi) - np.pi
            def to_relative_action(actions, robot_obs, max_pos=1, max_orn=1):
                rel_pos = actions[:3] - robot_obs[:3]
                rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
                rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
                rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
                gripper = actions[-1:]
                return np.concatenate([rel_pos, rel_orn, gripper])
            def quate2euler(quat, gimble_fix=False):
                quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
                if quat[-1] < 0: quat = -quat
                euler = pybullet.getEulerFromQuaternion(quat)
                return euler
            r_euler = np.array(quate2euler(frames[file_idx].gripper_pose[3:7]))
            a_euler = np.array(quate2euler(frames[file_idx+1].gripper_pose[3:7]))
            # r_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx].gripper_pose[3:7]))
            # a_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx+1].gripper_pose[3:7]))
            if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8: # 大于0.8就发生了万向节死锁
                sr_euler = np.array(scipyR.from_quat(frames[file_idx].gripper_pose[3:7]).as_euler('xyz', degrees=False))
                sa_euler = np.array(scipyR.from_quat(frames[file_idx+1].gripper_pose[3:7]).as_euler('xyz', degrees=False))
                if np.abs(angle_between_angles(a_euler, r_euler) - angle_between_angles(sa_euler, sr_euler)).max()<0.01: # 并没有发生万向节死锁
                    continue
                elif np.abs(angle_between_angles(sa_euler, sr_euler)).max() < 0.8: # 大于0.8就发生了万向节死锁
                    r_euler = sr_euler
                    a_euler = sa_euler
                else:
                    if np.abs(angle_between_angles(sa_euler, r_euler)).max() < 0.8:
                        a_euler = sa_euler
                    elif np.abs(angle_between_angles(a_euler, sr_euler)).max() < 0.8:
                        r_euler = sr_euler
                    else:
                        continue
            robot_obs_euler = np.concatenate([frames[file_idx].gripper_pose[:3], r_euler, [frames[file_idx].gripper_open]]) # YF: gripper_open=0/1, 注意要求的是-1和1
            actions_euler = np.concatenate([frames[file_idx+1].gripper_pose[:3], a_euler, [frames[file_idx+1].gripper_open]])
            action = to_relative_action(actions_euler, robot_obs_euler[:6])
            
            def calculate_fov(intrinsic_matrix, image_width):
                fx = intrinsic_matrix[0, 0]
                fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
                return fov
            rgb2 = np.array(Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png"))
            ep = {
                 "calib_gripper": {"extrinsic_matrix": frames[file_idx].misc['wrist_camera_extrinsics'],
                                    "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                    "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                    "cam_config": {"width": rgb2.shape[1], "height": rgb2.shape[0], "fov": calculate_fov(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.shape[0]), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}},

            }
            gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
            loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix),np.array([0,0,0,1]))
            gripper_locs.append(loc)
            actions.append(action)
            
            # if action[3:6].min() < -0.1 or action[3:6].max() > 0.1:
            #     print(robot_obs_euler, actions_euler)
            #     rgb = np.array(Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png"))
            #     rgb_ = np.array(Image.open(episodes_dir/f"wrist_rgb/{file_idx+1}.png"))
            #     cv2.imwrite(f"tmp/{task_str}_{num_demo}_{file_idx}_{np.array2string(action[3:6], precision=4, separator=',', suppress_small=True)}.jpg", np.hstack([rgb, rgb_])[...,::-1])
            if 0:
                rgb = np.array(Image.open(episodes_dir/f"front_rgb/{file_idx}.png"))
                rgb2 = np.array(Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png"))
                def calculate_fov(intrinsic_matrix, image_width):
                    fx = intrinsic_matrix[0, 0]
                    fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
                    return fov
                with open(episodes_dir / "front_depth" / f"{file_idx}.pkl",'rb') as f:
                    depth_static = pickle.load(f)
                with open(episodes_dir / "wrist_depth" / f"{file_idx}.pkl",'rb') as f:
                    depth_gripper = pickle.load(f)
                ep = {
                    "rgb_static": rgb,
                    "rgb_gripper": rgb2,
                    "depth_static": depth_static,
                    "depth_gripper": depth_gripper,
                    "rel_actions": action,
                    "robot_obs": frames[file_idx].joint_positions,
                    "scene_obs": frames[file_idx].task_low_dim_state,
                    "calib_static": {"extrinsic_matrix": frames[file_idx].misc['front_camera_extrinsics'],
                                    "intrinsic_matrix": frames[file_idx].misc['front_camera_intrinsics'],
                                    "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                    "cam_config": {"width": rgb.shape[1], "height": rgb.shape[0], "fov": calculate_fov(frames[file_idx].misc['front_camera_intrinsics'], rgb.shape[0]), "nearval": frames[file_idx].misc['front_camera_near'], "farval":  frames[file_idx].misc['front_camera_far']}},
                    "calib_gripper": {"extrinsic_matrix": frames[file_idx].misc['wrist_camera_extrinsics'],
                                    "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                    "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                    "cam_config": {"width": rgb2.shape[1], "height": rgb2.shape[0], "fov": calculate_fov(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.shape[0]), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}},
                }

                rgb, calib = {}, {}
                depth_static = ep['depth_static']
                depth_gripper = ep['depth_gripper']
                static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) 
                gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
                
                static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
                gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
                static_pcd = deproject(
                    static_cam, depth_static,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                gripper_pcd = deproject(
                    gripper_cam, depth_gripper,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

                rgb['rgb_static'] = Image.fromarray(ep['rgb_static']) #
                rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper']) # 
                rgb['rgb_static'] = np.array(rgb['rgb_static'])
                rgb['rgb_gripper'] = np.array(rgb['rgb_gripper'])
                static_rgb =  np.reshape(
                    rgb['rgb_static'], ( rgb['rgb_static'].shape[0]*rgb['rgb_static'].shape[1], 3)
                )
                gripper_rgb =  np.reshape(
                    rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0]*rgb['rgb_gripper'].shape[1], 3)
                )
                pcd_rgb = np.concatenate([static_rgb,gripper_rgb],axis=0)
                pcd_rgb = pcd_rgb/255
                if 1:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
                    o3d.io.write_point_cloud("tmp.pcd", pcd)
                    pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
                    o3d.io.write_point_cloud("tmp1.pcd", pcd)

gripper_locs = np.stack(gripper_locs)
actions=np.stack(actions)
print(gripper_locs.min(0), gripper_locs.max(0))
print(actions.min(0), actions.max(0))
np.save('data_folder/colosseum_gripper_limit.npy', gripper_locs)
np.save('data_folder/colosseum_actions.npy', actions)
for i in range(3):
    print(plt.hist(gripper_locs[:, i], bins=20, color='blue', edgecolor='black'))
for i in range(7):
    print(plt.hist(actions[:, i], bins=20, color='blue', edgecolor='black'))
