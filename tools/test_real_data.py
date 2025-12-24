import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

#ee_pose = np.load("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/episode_0000002/ee_pose.npy")


# ~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/arm_pose.txt

# 读取图片
# arm_depth = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/gripper_depth.png',cv2.IMREAD_UNCHANGED)
# arm_depth = arm_depth/1000
# arm_rgb = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/gripper_rgb.png')
# base_depth = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/base_depth.png',cv2.IMREAD_UNCHANGED)
# base_depth = base_depth/1000
# base_rgb = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/base_rgb.png')
# base_rgb = base_rgb[:, :, ::-1]
# arm_rgb = arm_rgb[:, :, ::-1]
# arm_i = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/gripper_intrinsic.txt")

# base_i = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/base_intrinsic.txt")

# base_e = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/eye_to_hand_pose.txt")

# arm_e = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/eye_in_hand_pose.txt")

# state = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/test_data/arm_pose.txt")


arm_depth = cv2.imread('~/yanfeng/project/robotic/robo_mm/tmp/test_data/gripper_depth.png',cv2.IMREAD_UNCHANGED)
arm_depth = arm_depth/1000
arm_rgb = cv2.imread('~/yanfeng/project/robotic/robo_mm/tmp/test_data/gripper_rgb.png')
base_depth = cv2.imread('~/yanfeng/project/robotic/robo_mm/tmp/test_data/base2_depth.png',cv2.IMREAD_UNCHANGED)
base_depth = base_depth/1000
base_rgb = cv2.imread('~/yanfeng/project/robotic/robo_mm/tmp/test_data/base2_rgb.png')
base_rgb = base_rgb[:, :, ::-1]
arm_rgb = arm_rgb[:, :, ::-1]
arm_i = np.loadtxt("~/yanfeng/project/robotic/robo_mm/tmp/test_data/gripper_intrinsic.txt")

base_i = np.loadtxt("~/yanfeng/project/robotic/robo_mm/tmp/test_data/base2_intrinsic.txt")

base_e = np.loadtxt("~/yanfeng/project/robotic/robo_mm/tmp/test_data/eye_to_hand_pose2.txt")

arm_e = np.loadtxt("~/yanfeng/project/robotic/robo_mm/tmp/test_data/eye_in_hand_pose.txt")

state = np.loadtxt("~/yanfeng/project/robotic/robo_mm/tmp/test_data/arm_pose.txt")

# arm_depth = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/arm_depth.png',cv2.IMREAD_UNCHANGED)
# arm_depth = arm_depth/1000
# arm_rgb = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/arm_rgb.png')
# base_depth = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/base_depth.png',cv2.IMREAD_UNCHANGED)
# base_depth = base_depth/1000
# base_rgb = cv2.imread('~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/base_rgb.png')
# base_rgb = base_rgb[:, :, ::-1]
# arm_rgb = arm_rgb[:, :, ::-1]
# arm_i = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/arm_intrinsic.txt")

# base_i = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/base_intrinsic.txt")

# base_e = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/eye_to_hand_pose.txt")

# arm_e = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/eye_in_hand_pose.txt")

# state = np.loadtxt("~/liufanfan/workspace/RoboUniView_3_real/robouniview/data/real_data/arm_pose.txt")




arm_pose_mat = np.eye(4)
arm_pose_mat[:3, 3] = state[:3]
arm_pose_mat[:3, :3] = R.from_euler('xyz', state[3:], False).as_matrix()

arm_e = np.matmul(arm_pose_mat, arm_e)

def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix))
    z = depth_img[v, u]
    #foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    foc = cam.foc
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


class cam:
    """Point object class for keypoint detection"""

    def __init__(self,viewMatrix,height,width,foc):
        self.viewMatrix = viewMatrix
        self.height = height
        self.width = width
        self.foc = foc

# transformation_matrix = pose_to_transformation_matrix(state[:3],state[3:])
# arm_e = np.dot(transformation_matrix,arm_e)
arm_e = np.linalg.inv(arm_e)
base_e = np.linalg.inv(base_e)

arm_e = arm_e*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
base_e = base_e*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])


static_cam = cam(base_e,base_i[1,2]*2,base_i[0,2]*2,base_i[0,0])
gripper_cam = cam(arm_e,arm_i[1,2]*2,arm_i[0,2]*2,arm_i[0,0])





static_pcd = deproject(
                static_cam, base_depth,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)

gripper_pcd = deproject(
            gripper_cam, arm_depth,
            homogeneous=False, sanity_check=False
        ).transpose(1, 0)


print(static_pcd)
print(gripper_pcd)

cloud = np.concatenate([static_pcd,gripper_pcd],axis=0)

static_rgb =  np.reshape(
    base_rgb, (480*640,3)
)/255
gripper_rgb =  np.reshape(
    arm_rgb, (480*640,3)
)/255
pcd_rgb = np.concatenate([static_rgb,gripper_rgb],axis=0)
#pcd_rgb = pcd_rgb/255


# voxel_range = [[-1, 1], [-1, 1], [-1, 1]]
# voxel_range = np.array(voxel_range)
# mask = np.logical_and(cloud > voxel_range[:, 0], cloud < voxel_range[:, 1])
# mask = mask.all(axis=-1)
# cloud = cloud[mask]
# pcd_rgb = pcd_rgb[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
o3d.io.write_point_cloud("4.pcd", pcd)


voxel_range = [[-2, 2], [-2, 2], [-2, 2]]
voxel_range = np.array(voxel_range)
mask = np.logical_and(static_pcd > voxel_range[:, 0], static_pcd < voxel_range[:, 1])
mask = mask.all(axis=-1)
static_pcd = static_pcd[mask]
static_rgb = static_rgb[mask]

static_pcd = np.concatenate([static_pcd,np.array([[0,0,0],[0,0,0.05],[0,0,-0.05]])],axis=0)
static_rgb = np.concatenate([static_rgb,np.array([[0.9,0,0],[0.9,0,0],[0.9,0,0]])],axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
pcd.colors = o3d.utility.Vector3dVector(static_rgb)
o3d.io.write_point_cloud("6.pcd", pcd)


# voxel_range = [[-1, 1], [-1, 1], [-1, 1]]
# voxel_range = np.array(voxel_range)
# mask = np.logical_and(gripper_pcd > voxel_range[:, 0], gripper_pcd < voxel_range[:, 1])
# mask = mask.all(axis=-1)
# gripper_pcd = gripper_pcd[mask]
# gripper_rgb = gripper_rgb[mask]


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
pcd.colors = o3d.utility.Vector3dVector(gripper_rgb)
o3d.io.write_point_cloud("5.pcd", pcd)
#pcd = self.vfe_generator.generate(np.asarray(pcd.points),np.asarray(pcd.colors))
#pcd = self.vfe_generator.generate(cloud[:, :3],pcd_rgb)



print("done!!")