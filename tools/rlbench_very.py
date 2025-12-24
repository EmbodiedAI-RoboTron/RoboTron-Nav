
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode
def create_obs_config(
    image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
):
    """
    Set up observation config for RLBench environment.
        :param image_size: Image size.
        :param apply_rgb: Applying RGB as inputs.
        :param apply_depth: Applying Depth as inputs.
        :param apply_pc: Applying Point Cloud as inputs.
        :param apply_cameras: Desired cameras.
        :return: observation config
    """
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=apply_rgb,
        point_cloud=apply_pc,
        depth=apply_depth,
        mask=False,
        image_size=image_size,
        render_mode=RenderMode.OPENGL,
        **kwargs,
    )

    camera_names = apply_cameras
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams

    obs_config = ObservationConfig(
        front_camera=kwargs.get("front", unused_cams),
        left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        wrist_camera=kwargs.get("wrist", unused_cams),
        overhead_camera=kwargs.get("overhead", unused_cams),
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config
# setup RLBench environments
image_size, apply_rgb, apply_depth, apply_pc, apply_cameras = (128, 128), True, True, True, ("left_shoulder", "right_shoulder", "wrist", "front")
obs_config = create_obs_config(
    image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
)
collision_checking, headless = False, False
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
    gripper_action_mode=Discrete()
)
env = Environment(
    action_mode, str("~/robotics/RLbench/peract/raw/test/"), obs_config,
    headless=headless
)
task_name, variation, episode_index = "place_cups", 0, 0
demos = env.get_demos(
    task_name=task_name,
    variation_number=variation,
    amount=1,
    from_episode_number=episode_index,
    random_selection=False
)

from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, get_gripper_camera_view_matrix, RandomShiftsAug, deproject_metaworld
static_cam = cam(static_extrinsic_matrix, ep['calib_static']['cam_config']['height'], ep['calib_static']['cam_config']['width'], ep['calib_static']['cam_config']['fov'])


def pointcloud_from_depth_and_camera_params(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in world frame.
    :return: A numpy array of size (width, height, 3)
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()
    
    # 获取焦距和相机中心点
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 深度值
    z = depth.ravel()
    
    # 计算相机坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    z = z  # 保持为正值
    
    # 将相机坐标转换为齐次坐标
    cam_coords = np.vstack((x, y, z, np.ones_like(z)))

    # 计算视图矩阵的逆
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics_ = np.concatenate((R_inv, -R_inv_C), -1)
    extrinsics_homo = np.concatenate([extrinsics_, [np.array([0, 0, 0, 1])]])
    extrinsics_inv_homo = np.linalg.inv(extrinsics_homo)
    # 将相机坐标转换为世界坐标
    world_coords_homo = extrinsics_inv_homo @ cam_coords
    
    # 如果不需要齐次坐标，去掉最后一行
    world_coords = world_coords_homo[:3, :].reshape((3, h, w)).transpose((1, 2, 0))
    
    return world_coords


def pointcloud_from_depth_and_camera_params(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in world frame.
    :return: A numpy array of size (width, height, 3)
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()
    
    # 计算视图矩阵的逆
    # C = np.expand_dims(extrinsics[:3, 3], 0).T
    # R = extrinsics[:3, :3]
    # R_inv = R.T  # inverse of rot matrix is transpose
    # R_inv_C = np.matmul(R_inv, C)
    # extrinsics_ = np.concatenate((R_inv, -R_inv_C), -1)
    # extrinsics_homo = np.concatenate([extrinsics_, [np.array([0, 0, 0, 1])]])
    # extrinsics_inv_homo = np.linalg.inv(extrinsics_homo)
    T_world_cam = extrinsics
    T_world_cam = np.linalg.inv(np.array(static_cam.viewMatrix))
    z = depth[v, u]
    foc = static_cam.height / (2 * np.tan(np.deg2rad(static_cam.fov) / 2))
    # 计算相机坐标
    x = (u - static_cam.width // 2) * z / foc
    y = -(v - static_cam.height // 2) * z / foc
    z = -z  # 保持为正值
    ones = np.ones_like(z)
    # 将相机坐标转换为齐次坐标
    cam_pos = np.stack([x, y, z, ones], axis=0)

    world_coords = T_world_cam @ cam_pos
    
    return world_coords