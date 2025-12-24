import io, os, sys
import numpy as np
from PIL import Image
import open3d as o3d
import zipfile, pickle
import matplotlib.pyplot as plt
from robouniview.data.zipreader import ZipReader   
from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
from concurrent.futures import ThreadPoolExecutor


abs_datasets_dir = '~/yanfeng/data/robotics/libero_uniview_final'
_task_list = os.listdir(abs_datasets_dir)

if 1:
    def process_file(file_path):

        def rotMatList2NPRotMat(rot_mat_arr):
            np_rot_arr = np.array(rot_mat_arr)
            np_rot_mat = np_rot_arr.reshape((3, 3))
            return np_rot_mat
        def posRotMat2Mat(pos, rot_mat):
            t_mat = np.eye(4)
            t_mat[:3, :3] = rot_mat
            t_mat[:3, 3] = np.array(pos)
            return t_mat

        episode = np.load(file_path, allow_pickle=True)
        action = episode['action']

        base_extrinsic = posRotMat2Mat(episode['base_p'], rotMatList2NPRotMat(episode['base_r']))
        calib2 = episode['robot0_eye_in_hand'].item()['calib']
        ep = {
            "calib_gripper": calib2,
        }
        gripper_extrinsic_matrix = np.linalg.inv(ep['calib_gripper']['extrinsic_matrix']) * np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
        gripper_loc = np.dot(np.linalg.inv(gripper_extrinsic_matrix), np.array([0,0,0,1]))

        if 1:
            ep = {
                "rgb_static": episode['agentview'].item()['image'],
                "rgb_gripper": episode['robot0_eye_in_hand'].item()['image'],
                "rel_actions": action,
                "robot_obs": np.zeros(20), # 由于没有保存，且没有使用，直接复制为0了。
                "scene_obs": np.zeros(20),
                "depth_static": episode['agentview'].item()['depth'],
                "depth_gripper": episode['robot0_eye_in_hand'].item()['depth'],
                "calib_static": episode['agentview'].item()['calib'],
                "calib_gripper": episode['robot0_eye_in_hand'].item()['calib'],
                "base_pose": posRotMat2Mat(episode['base_p'], rotMatList2NPRotMat(episode['base_r']))
            }

            rgb, calib = {}, {}
            depth_static = ep['depth_static']
            depth_gripper = ep['depth_gripper']
            static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
            gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
            # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
            static_extrinsic_matrix = np.dot(static_extrinsic_matrix, ep["base_pose"])
            gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, ep["base_pose"])
            static_cam = cam(static_extrinsic_matrix, 256, 256, 45)
            gripper_cam = cam(gripper_extrinsic_matrix, 256, 256, 75)

            static_pcd = deproject(
                static_cam, depth_static,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = deproject(
                gripper_cam, depth_gripper,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)

            rgb['rgb_static'] = Image.fromarray(ep['rgb_static']) # 注意此处需要添加flip
            rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper']) # 注意此处需要添加flip
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
                pcd.points = o3d.utility.Vector3dVector(cloud)
                pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
                o3d.io.write_point_cloud("tmp.pcd", pcd)

        return action, gripper_loc

    def process_task(task, abs_datasets_dir):
        data_dir = os.path.join(abs_datasets_dir, task)
        if not os.path.isdir(data_dir): return  # 样本均在文件夹内，假如非文件夹则无效
        demo_num = len(os.listdir(data_dir))

        for demo_id in range(demo_num):
            demo_path = os.path.join(data_dir, f"demo_{demo_id}")
            if not os.path.isdir(demo_path):
                continue

            files = [os.path.join(demo_path, f"{file_idx}.npz") for file_idx in range(len(os.listdir(demo_path)))]
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = list(executor.map(process_file, files))

            for action, gripper_loc in results:
                actions.append(action)
                gripper_locs.append(gripper_loc)
    
    gripper_locs = []
    actions = []
    for task in _task_list:
        process_task(task, abs_datasets_dir)

    gripper_locs=np.stack(gripper_locs)
    print(gripper_locs.min(0), gripper_locs.max(0))
    np.save('data_folder/libero_gripper_limit.py', gripper_locs)
    for i in range(3):
        print(plt.hist(gripper_locs[:,i], bins=20, color='blue', edgecolor='black'))

    actions=np.stack(actions)
    print(actions.min(0), actions.max(0))
    np.save('data_folder/libero_actions.py', gripper_locs)
    for i in range(7):
        print(plt.hist(actions[:,i], bins=20, color='blue', edgecolor='black'))



def show(env, filename):
    def rotMatList2NPRotMat(rot_mat_arr):
        np_rot_arr = np.array(rot_mat_arr)
        np_rot_mat = np_rot_arr.reshape((3, 3))
        return np_rot_mat
    def quat2Mat(quat):
        if len(quat) != 4:
            print("Quaternion", quat, "invalid when generating transformation matrix.")
            raise ValueError

        # Note that the following code snippet can be used to generate the 3x3
        #    rotation matrix, we don't use it because this file should not depend
        #    on mujoco.
        '''
        from mujoco_py import functions
        res = np.zeros(9)
        functions.mju_quat2Mat(res, camera_quat)
        res = res.reshape(3,3)
        '''

        # This function is lifted directly from scipy source code
        #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
            2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
            2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
        np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
        return np_rot_mat
    def posRotMat2Mat(pos, rot_mat):
        t_mat = np.eye(4)
        t_mat[:3, :3] = rot_mat
        t_mat[:3, 3] = np.array(pos)
        return t_mat
    calib = {}
    for view_map in [["agentview", "rgb_static"], ["robot0_eye_in_hand", "rgb_gripper"]]:
        view = view_map[0] # corner2
        cam_id = env.sim.model.camera_name2id(view)
        fov = env.sim.model.cam_fovy[cam_id]
        # 使用动态外参
        cam_pos = env.sim.data.cam_xpos[cam_id]
        c2b_r = rotMatList2NPRotMat(env.sim.data.cam_xmat[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        static_extrinsic = posRotMat2Mat(cam_pos, c2w_r)
        static_foc = resolution[0] / (2 * np.tan(np.deg2rad(fov) / 2))
        static_intrinsic = np.array([[static_foc , 0.0, resolution[1]/2], [0.0, static_foc , resolution[0]/2], [0.0, 0.0, 1.0]])
        cam_config = {'_target_': 'calvin_env.camera.static_camera.StaticCamera', 'name': 'static', 'fov': fov, 'aspect': 1, 
            'nearval': env.sim.model.vis.map.znear, 'farval': env.sim.model.vis.map.zfar, 'extent': env.sim.model.stat.extent, 'width': resolution[1], 'height': resolution[0], 
        }
        calib[view_map[1]] = {'extrinsic_matrix':static_extrinsic,
                            'intrinsic_matrix':static_intrinsic,
                            'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                            'cam_config': cam_config}
    
    static_extrinsic_matrix = np.linalg.inv(calib['rgb_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    gripper_extrinsic_matrix = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
    base_p, base_r = env.sim.data.body_xpos[2], env.sim.data.body_xmat[2] # 2是底盘的id，其他仿真器需要查一下body_name
    base_r = rotMatList2NPRotMat(base_r)
    base_extrinsic = posRotMat2Mat(base_p, base_r)
    # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
    static_extrinsic_matrix = np.dot(static_extrinsic_matrix, base_extrinsic)
    gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, base_extrinsic)
    
    # 平移矩阵 T_translate
    T_translate = np.array([
        [1, 0, 0, 0.3],    # 
        [0, 1, 0, 0.0],     # 
        [0, 0, 1, -0.1],    # [-0.24398557 -0.43166057  0.01677953  1.        ] [0.86626239 0.57459427 0.90260449 1.        ]  
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

    if 1:
        from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
        static_cam = cam(static_extrinsic_matrix, calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width'], calib['rgb_static']['cam_config']['fov'])
        gripper_cam = cam(gripper_extrinsic_matrix,  calib['rgb_gripper']['cam_config']['height'],  calib['rgb_gripper']['cam_config']['width'],  calib['rgb_gripper']['cam_config']['fov'])
        def depthimg2Meters(depth, cam):
            extent =cam['cam_config']['extent']
            near = cam['cam_config']['nearval'] * extent
            far = cam['cam_config']['farval'] * extent
            image = near / (1 - depth * (1 - near / far))
            return image
        rgb_static, depth_static = env.sim.render(*resolution, camera_name="agentview", depth=True)
        rgb_gripper, depth_gripper = env.sim.render(*resolution, camera_name="robot0_eye_in_hand", depth=True)
        depth_static = np.flip(depth_static, axis=0) # 注意训练集中depth和rgb已经flip过了
        depth_static = depthimg2Meters(depth_static, calib['rgb_static'])
        depth_gripper = np.flip(depth_gripper, axis=0)
        depth_gripper = depthimg2Meters(depth_gripper, calib['rgb_gripper'])
        
        static_pcd = deproject(
            static_cam, depth_static,
            homogeneous=False, sanity_check=False
        ).transpose(1, 0)
        gripper_pcd = deproject(
            gripper_cam, depth_gripper,
            homogeneous=False, sanity_check=False
        ).transpose(1, 0)
        rgb_static = np.flip(rgb_static, axis=0).reshape(-1, 3)/255. # 注意obs["rgb_obs"]['rgb_gripper']已经在上一个函数翻转过了
        rgb_gripper = np.flip(rgb_gripper, axis=0).reshape(-1, 3)/255.

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_static)
        o3d.io.write_point_cloud(f"tmp/{filename}0.pcd", pcd)
        pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_gripper)
        o3d.io.write_point_cloud(f"tmp/{filename}1.pcd", pcd)


# def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
#     """
#     Deprojects a pixel point to 3D coordinates
#     Args
#         point: tuple (u, v); pixel coordinates of point to deproject
#         depth_img: np.array; depth image used as reference to generate 3D coordinates
#         homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
#                      else returns the world coordinates (x, y, z) position
#     Output
#         (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
#     """
#     h, w = depth_img.shape
#     u, v = np.meshgrid(np.arange(w), np.arange(h))
#     u, v = u.ravel(), v.ravel()

#     # Unproject to world coordinates
#     T_world_cam = np.linalg.inv(np.array(cam.viewMatrix))
#     z = depth_img[v, u]
#     foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
#     x = (u - cam.width // 2) * z / foc
#     y = -(v - cam.height // 2) * z / foc
#     z = -z
#     ones = np.ones_like(z)

#     cam_pos = np.stack([x, y, z, ones], axis=0)
#     world_pos = T_world_cam @ cam_pos

#     # Sanity check by using camera.deproject function.  Check 2000 points.

#     if not homogeneous:
#         world_pos = world_pos[:3]

#     return world_pos


# for task in _task_list:
#     data_dir = os.path.join(abs_datasets_dir, task)
#     if not os.path.isdir(data_dir): continue # 样本均在文件夹内，假如非文件夹则无效
#     demo_num = len(os.listdir(data_dir))
#     for demo_i in range(demo_num):
#         demo_path = os.path.join(data_dir, f"demo_{demo_i}")
#         for file_idx in range(len(os.listdir(demo_path))):
#             file_path = os.path.join(demo_path, f"{file_idx}.npz")
#             episode = np.load(file_path, allow_pickle=True)
#             def rotMatList2NPRotMat(rot_mat_arr):
#                 np_rot_arr = np.array(rot_mat_arr)
#                 np_rot_mat = np_rot_arr.reshape((3, 3))
#                 return np_rot_mat
#             def posRotMat2Mat(pos, rot_mat):
#                 t_mat = np.eye(4)
#                 t_mat[:3, :3] = rot_mat
#                 t_mat[:3, 3] = np.array(pos)
#                 return t_mat

#             ep= {
#                 "rgb_static": episode['agentview'].item()['image'],
#                 "rgb_gripper": episode['robot0_eye_in_hand'].item()['image'],
#                 "rel_actions": episode['action'],
#                 "robot_obs": np.zeros(20), # 由于没有保存，且没有使用，直接复制为0了。
#                 "scene_obs": np.zeros(20),
#                 "depth_static": episode['agentview'].item()['depth'],
#                 "depth_gripper": episode['robot0_eye_in_hand'].item()['depth'],
#                 "calib_static": episode['agentview'].item()['calib'],
#                 "calib_gripper": episode['robot0_eye_in_hand'].item()['calib'],
#                 "base_pose": posRotMat2Mat(episode['base_p'], rotMatList2NPRotMat(episode['base_r']))
#             }

#             rgb, calib = {}, {}
#             depth_static = ep['depth_static']
#             depth_gripper = ep['depth_gripper']
#             # # 所有外参表示从XX到世界的RT, C=A^-1*B 摄像头到世界坐标系的RT关系A和机器人到世界坐标系的RT关系B, C为摄像头到机器人的RT
#             # static_extrinsic_matrix = np.dot(np.linalg.inv(ep["base_pose"]), ep['calib_static']['extrinsic_matrix'])
#             # static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix)*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])

#             static_extrinsic_matrix = np.linalg.inv(ep['calib_static']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
#             gripper_extrinsic_matrix = np.linalg.inv( ep['calib_gripper']['extrinsic_matrix'])*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
#             # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
#             static_extrinsic_matrix = np.dot(static_extrinsic_matrix, ep["base_pose"])
#             gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, ep["base_pose"])

#             T_translate = np.array([
#                 [1, 0, 0, 0.45],    # 
#                 [0, 1, 0, 0.0],     # 
#                 [0, 0, 1, -0.45],    # [-0.20068789 -0.40613651  0.06949247  1.        ] [0.83878395 0.49946681 0.8438832  1.      
#                 [0, 0, 0, 1] 
#             ])
#             static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
#             gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐

#             gripper_locs.append(np.dot(np.linalg.inv(gripper_extrinsic_matrix),np.array([0,0,0,1])))
#             static_cam = cam(static_extrinsic_matrix, 256, 256, 45)
#             gripper_cam = cam(gripper_extrinsic_matrix, 256, 256, 75)

#             static_pcd = deproject(
#                 static_cam, depth_static,
#                 homogeneous=False, sanity_check=False
#             ).transpose(1, 0)
#             gripper_pcd = deproject(
#                 gripper_cam, depth_gripper,
#                 homogeneous=False, sanity_check=False
#             ).transpose(1, 0)
#             cloud = np.concatenate([static_pcd, gripper_pcd],axis=0)
            
#             rgb['rgb_static'] = Image.fromarray(ep['rgb_static']) # 注意此处需要添加flip
#             rgb['rgb_gripper'] =  Image.fromarray(ep['rgb_gripper']) # 注意此处需要添加flip
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
