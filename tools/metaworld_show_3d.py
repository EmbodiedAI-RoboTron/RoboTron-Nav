import io, os, json
import numpy as np
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld



abs_datasets_dir = '~/yanfeng/data/robotics/metaworld_uniview/'
json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
with open(json_path, 'r') as f:
    data = json.load(f)
if 1:
    gripper_locs = []
    actions=[]
    for data_slice in data:
        zip_file = data_slice[0]
        zip_file = os.path.join(abs_datasets_dir, zip_file)

        # zip_file = "~/yanfeng/data/robotics/robocasa_uniview/single_stage_kitchen_drawer_CloseDrawer_2024-04-30/demo_1.zip"
        # print(zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            with zip_ref.open('param/param.pickle') as file:
                with io.BytesIO(file.read()) as file_like_object:
                    frames = pickle.load(file_like_object)
        # base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
        # def rotMatList2NPRotMat(rot_mat_arr):
        #     np_rot_arr = np.array(rot_mat_arr)
        #     np_rot_mat = np_rot_arr.reshape((3, 3))
        #     return np_rot_mat
        # def posRotMat2Mat(pos, rot_mat):
        #     t_mat = np.eye(4)
        #     t_mat[:3, :3] = rot_mat
        #     t_mat[:3, 3] = np.array(pos)
        #     return t_mat
        # base_r = rotMatList2NPRotMat(base_r)
        # base_extrinsic = posRotMat2Mat(base_p, base_r)

        for file_idx in range(len(frames)):
            actions.append(frames[file_idx]['action'])

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                with zip_ref.open(os.path.join('rgb', "behindGripper", 'config_save_' + str(file_idx) + ".npy")) as file:
                    with io.BytesIO(file.read()) as file_like_object:
                        calib2 = np.load(file_like_object, allow_pickle=True).item()
            ep = {
                "calib_gripper": calib2,
            }
            gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
           
            loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix),np.array([0,0,0,1]))
            gripper_locs.append(loc)

            if 1:
                rgb = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "corner2", 'image_save_' + str(file_idx) + ".jpg"))))
                rgb2 = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "behindGripper", 'image_save_' + str(file_idx) + ".jpg"))))
                rgb_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "corner2", 'depth_save_' + str(file_idx) + ".npy"))))
                rgb2_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "behindGripper", 'depth_save_' + str(file_idx) + ".npy"))))
                calib = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "corner2", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                calib2 = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "behindGripper", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
                ep = {
                    "rgb_static": np.array(rgb),
                    "rgb_gripper": np.array(rgb2),
                    # "rel_actions": action,  # action_scale: float = 1.0 / 100
                    "robot_obs": frames[file_idx]['obs'],
                    "scene_obs": frames[file_idx]['obs'],
                    "depth_static": rgb_depth,
                    "depth_gripper": rgb2_depth,
                    "calib_static": calib,
                    "calib_gripper": calib2,
                }

                rgb, calib = {}, {}
                depth_static = ep['depth_static']
                depth_gripper = ep['depth_gripper']
                static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
                gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
                
                static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
                gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
                def depthimg2Meters(depth, cam):
                    extent =cam['cam_config']['extent']
                    near = cam['cam_config']['nearval'] * extent
                    far = cam['cam_config']['farval'] * extent
                    image = near / (1 - depth * (1 - near / far))
                    return image
                depth_static = np.flip(depth_static, axis=0)
                depth_static = depthimg2Meters(depth_static, ep['calib_static'])
                depth_gripper = np.flip(depth_gripper, axis=0)
                depth_gripper = depthimg2Meters(depth_gripper, ep['calib_gripper'])
                
                static_pcd = deproject(
                    static_cam, depth_static,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                gripper_pcd = deproject(
                    gripper_cam, depth_gripper,
                    homogeneous=False, sanity_check=False
                ).transpose(1, 0)
                cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

                rgb['rgb_static'] = Image.fromarray(np.flip(ep['rgb_static'], axis=0)) # 注意此处需要添加flip
                rgb['rgb_gripper'] =  Image.fromarray(np.flip(ep['rgb_gripper'], axis=0)) # 注意此处需要添加flip
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
                    _pcd = o3d.geometry.PointCloud()
                    _pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    _pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                    o3d.io.write_point_cloud("tmp.pcd", _pcd)

    gripper_locs=np.stack(gripper_locs)
    np.save('data_folder/netaworld_gripper_limit.py', gripper_locs)
    print(gripper_locs.min(0), gripper_locs.max(0))
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))
    # [-0.5038039  -0.10849035  0.12061001  1.        ] [0.48056049 0.41590198 0.60125949 1.        ]
    actions=np.stack(actions)
    print(actions.min(0), actions.max(0))
    np.save('data_folder/metaworld_actions.py', gripper_locs)
    for i in range(4):
        print(plt.hist(actions[:,i], bins=20, color='blue', edgecolor='black'))

# for task in _task_list:
#     data_dir = os.path.join(abs_datasets_dir, task)
#     if not os.path.isdir(data_dir): continue # 样本均在文件夹内，假如非文件夹则无效
#     for demo_name in [tmp for tmp in os.listdir(data_dir) if '.zip' in tmp]:
#         zip_file = os.path.join(data_dir, demo_name)
#         # zip_file = "~/yanfeng/data/robotics/robocasa_uniview/single_stage_kitchen_drawer_CloseDrawer_2024-04-30/demo_1.zip"
#         print(zip_file)
#         frames = pickle.load(io.BytesIO(ZipReader.read(zip_file, 'param/param.pickle')))
#         for file_idx in range(len(frames)):
#         # file_idx = 0
#             # ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
#             rgb = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'image_save_' + str(file_idx) + ".jpg"))))
#             rgb2 = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg"))))
#             rgb_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'depth_save_' + str(file_idx) + ".npy"))))
#             rgb2_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy"))))
#             calib = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
#             calib2 = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
#             ep = {
#                 "rgb_static": np.array(rgb),
#                 "rgb_gripper": np.array(rgb2),
#                 "depth_static": rgb_depth,
#                 "depth_gripper": rgb2_depth,
#                 "calib_static": calib,
#                 "calib_gripper": calib2,
#             }

#             env_param=np.load(io.BytesIO(ZipReader.read(zip_file, 'param/env.npy')), allow_pickle=True).item()
#             base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
#             def rotMatList2NPRotMat(rot_mat_arr):
#                 np_rot_arr = np.array(rot_mat_arr)
#                 np_rot_mat = np_rot_arr.reshape((3, 3))
#                 return np_rot_mat
#             def posRotMat2Mat(pos, rot_mat):
#                 t_mat = np.eye(4)
#                 t_mat[:3, :3] = rot_mat
#                 t_mat[:3, 3] = np.array(pos)
#                 return t_mat
#             base_r = rotMatList2NPRotMat(base_r)
#             base_extrinsic = posRotMat2Mat(base_p, base_r)


#             rgb, calib = {}, {}
#             depth_static = ep['depth_static']
#             depth_gripper = ep['depth_gripper']
#             static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
#             gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
#             # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
#             static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
#             gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)

#             # 平移矩阵 T_translate
#             T_translate = np.array([
#                 [1, 0, 0, 0.45], # [0.1, 0.8] - [-0.5, 0.5]
#                 [0, 1, 0, 0.0], # [-0.8, 0.8]
#                 [0, 0, 1, 0.6], # [0.9. 1.5]
#                 [0, 0, 0, 1]
#             ])
#             static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
#             gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐

#             gripper_locs.append(np.dot(np.linalg.inv(gripper_extrinsic_matrix),np.array([0,0,0,1])))

#             static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
#             gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
#             def depthimg2Meters(depth, cam):
#                 extent =cam['cam_config']['extent']
#                 near = cam['cam_config']['nearval'] * extent
#                 far = cam['cam_config']['farval'] * extent
#                 image = near / (1 - depth * (1 - near / far))
#                 return image
#             depth_static = np.flip(depth_static, axis=0)
#             depth_static = depthimg2Meters(depth_static, ep['calib_static'])
#             depth_gripper = np.flip(depth_gripper, axis=0)
#             depth_gripper = depthimg2Meters(depth_gripper, ep['calib_gripper'])
            
#             static_pcd = deproject(
#                 static_cam, depth_static,
#                 homogeneous=False, sanity_check=False
#             ).transpose(1, 0)
#             gripper_pcd = deproject(
#                 gripper_cam, depth_gripper,
#                 homogeneous=False, sanity_check=False
#             ).transpose(1, 0)
#             cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

#             rgb['rgb_static'] = Image.fromarray(np.flip(ep['rgb_static'], axis=0)) # 注意此处需要添加flip
#             rgb['rgb_gripper'] =  Image.fromarray(np.flip(ep['rgb_gripper'], axis=0)) # 注意此处需要添加flip        rgb['rgb_static'] = np.array(rgb['rgb_static'])
#             rgb['rgb_static'] = np.array(rgb['rgb_static'])
#             rgb['rgb_gripper'] = np.array(rgb['rgb_gripper'])

#             static_rgb =  np.reshape(
#                 rgb['rgb_static'], ( rgb['rgb_static'].shape[0]*rgb['rgb_static'].shape[1], 3)
#             )
#             gripper_rgb =  np.reshape(
#                 rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0]*rgb['rgb_gripper'].shape[1], 3)
#             )
#             pcd_rgb = np.concatenate([static_rgb,gripper_rgb],axis=0)
#             pcd_rgb = pcd_rgb/255
#             if 1:
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
#                 pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
#                 o3d.io.write_point_cloud(f"tmp/{task}_{demo_name}_{file_idx}.pcd", pcd)

#             # break

# gripper_locs=np.stack(gripper_locs)
# print(gripper_locs.min(0), gripper_locs.max(0))

# def rotMatList2NPRotMat(rot_mat_arr):
#     np_rot_arr = np.array(rot_mat_arr)
#     np_rot_mat = np_rot_arr.reshape((3, 3))
#     return np_rot_mat
# def quat2Mat(quat):
#     if len(quat) != 4:
#         print("Quaternion", quat, "invalid when generating transformation matrix.")
#         raise ValueError

#     # Note that the following code snippet can be used to generate the 3x3
#     #    rotation matrix, we don't use it because this file should not depend
#     #    on mujoco.
#     '''
#     from mujoco_py import functions
#     res = np.zeros(9)
#     functions.mju_quat2Mat(res, camera_quat)
#     res = res.reshape(3,3)
#     '''

#     # This function is lifted directly from scipy source code
#     #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
#     w = quat[0]
#     x = quat[1]
#     y = quat[2]
#     z = quat[3]

#     x2 = x * x
#     y2 = y * y
#     z2 = z * z
#     w2 = w * w

#     xy = x * y
#     zw = z * w
#     xz = x * z
#     yw = y * w
#     yz = y * z
#     xw = x * w

#     rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
#         2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
#         2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
#     np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
#     return np_rot_mat
# def posRotMat2Mat(pos, rot_mat):
#     t_mat = np.eye(4)
#     t_mat[:3, :3] = rot_mat
#     t_mat[:3, 3] = np.array(pos)
#     return t_mat

# if 0:
#     calibs=[]
#     for i in range(55):
#         zip_file=f'~/yanfeng/data/robotics/robocasa_uniview_tmp/single_stage_kitchen_coffee_CoffeePressButton_2024-04-25/demo_{i+1}.zip'
#         file_idx=0
#         # ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
#         rgb = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'image_save_' + str(file_idx) + ".jpg"))))
#         rgb2 = Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg"))))
#         rgb_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'depth_save_' + str(file_idx) + ".npy"))))
#         rgb2_depth = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy"))))
#         calib = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_agentview_right", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
#         calib2 = np.load(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy"))), allow_pickle=True).item()
#         env_param=np.load(io.BytesIO(ZipReader.read(zip_file, 'param/env.npy')), allow_pickle=True).item()
#         base_p, base_r = env_param['body_xpos'][1], env_param['body_xmat'][1]
#         base_r = rotMatList2NPRotMat(base_r)
#         base_extrinsic = posRotMat2Mat(base_p, base_r)
#         # calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], base_extrinsic)
#         # calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], base_extrinsic)
#         ep={
#             "rgb_static": np.array(rgb),
#             "rgb_gripper": np.array(rgb2),
#             "depth_static": rgb_depth,
#             "depth_gripper": rgb2_depth,
#             "calib_static": calib,
#             "calib_gripper": calib2,
#             }
#         calibs.append(calib)
        
#         depth_static = ep['depth_static']
#         depth_gripper = ep['depth_gripper']
#         static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
#         gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
#         static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
#         gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
#         T_translate = np.array([
#             [1, 0, 0, 0.45], # [0.1, 0.8] - [-0.5, 0.5]
#             [0, 1, 0, 0.0], # [-0.8, 0.8]
#             [0, 0, 1, 0.6], # [0.9. 1.5]
#             [0, 0, 0, 1]
#         ])
#         static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
#         gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
#         static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])
#         gripper_cam = cam(gripper_extrinsic_matrix,  ep['calib_gripper']['cam_config']['height'],  ep['calib_gripper']['cam_config']['width'],  ep['calib_gripper']['cam_config']['fov'])
#         def depthimg2Meters(depth, cam):
#             extent =cam['cam_config']['extent']
#             near = cam['cam_config']['nearval'] * extent
#             far = cam['cam_config']['farval'] * extent
#             image = near / (1 - depth * (1 - near / far))
#             return image
#         depth_static = np.flip(depth_static, axis=0)
#         depth_static = depthimg2Meters(depth_static, ep['calib_static'])
#         depth_gripper = np.flip(depth_gripper, axis=0)
#         depth_gripper = depthimg2Meters(depth_gripper, ep['calib_gripper'])
        
#         static_pcd = deproject(
#             static_cam, depth_static,
#             homogeneous=False, sanity_check=False
#         ).transpose(1, 0)
#         gripper_pcd = deproject(
#             gripper_cam, depth_gripper,
#             homogeneous=False, sanity_check=False
#         ).transpose(1, 0)
#         cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)
#         rgb={}
#         rgb['rgb_static'] = Image.fromarray(np.flip(ep['rgb_static'], axis=0)) # 注意此处需要添加flip
#         rgb['rgb_gripper'] =  Image.fromarray(np.flip(ep['rgb_gripper'], axis=0)) # 注意此处需要添加flip
#         rgb['rgb_static'] = np.array(rgb['rgb_static'])
#         rgb['rgb_gripper'] = np.array(rgb['rgb_gripper'])

#         static_rgb =  np.reshape(
#             rgb['rgb_static'], ( rgb['rgb_static'].shape[0]*rgb['rgb_static'].shape[1], 3)
#         )
#         gripper_rgb =  np.reshape(
#             rgb['rgb_gripper'], (rgb['rgb_gripper'].shape[0]*rgb['rgb_gripper'].shape[1], 3)
#         )
#         pcd_rgb = np.concatenate([static_rgb,gripper_rgb],axis=0)
#         pcd_rgb = pcd_rgb/255

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
#         pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
#         o3d.io.write_point_cloud(f"tmp/{i}.pcd", pcd)
#         # pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
#         # pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
#         # o3d.io.write_point_cloud(f"tmp/{i}.pcd", pcd)
