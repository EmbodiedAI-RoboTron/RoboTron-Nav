import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, time
from torch import Tensor, nn
from einops import rearrange, repeat

class cam:

    def __init__(self,viewMatrix,height,width,fov):
        self.viewMatrix = viewMatrix
        self.height = height
        self.width = width
        self.fov = fov
           
class PETR(nn.Module):
    def __init__(self,
                 hidden_dim = 1024, 
                 depth_step = 0.1,
                 depth_num = 60,
                 depth_start = 0.05,
                 position_range = [-0.5, -0.5, 0.3, 0.5, 0.5, 0.8],
                 ):
        super().__init__()
        #self.code_weights = self.code_weights[:self.code_size]
       #self.bg_cls_weight = 0
        self.embed_dims = hidden_dim
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start
        self.position_level = 0
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
 
            
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

      
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

   
    def position_embeding(self, img_feats, calibs, pad_h, pad_w, masks=None):

        with torch.no_grad():
                
            K_inv_all, extrinsics_inv_all = [], []
            for extrinsic, intrinsic, fov in zip(calibs['extrinsic_matrix'], calibs['intrinsic_matrix'], calibs['fov']): 
                fov, img_h, img_w = fov
                scale_x = pad_w / img_w
                scale_y = pad_h / img_h
                K = np.array(intrinsic.clone())
                K[0, 0] *= scale_x  # fx'
                K[1, 1] *= scale_y  # fy'
                K[0, 2] *= scale_x  # cx'
                K[1, 2] *= scale_y  # cy'
                # foc = pad_h / (2 * np.tan(np.deg2rad(fov) / 2))
                # K = np.array([
                #     [foc, 0, pad_w // 2],
                #     [0, foc, pad_h // 2],
                #     [0, 0, 1]
                # ])
                K_inv = np.linalg.inv(K)
                viewMatrix = np.array(extrinsic)
                extrinsics_inv = np.linalg.inv(viewMatrix)
                K_inv_all.append(torch.from_numpy(K_inv).to(img_feats.device).to(torch.float32))
                extrinsics_inv_all.append(torch.from_numpy(extrinsics_inv).to(img_feats.device).to(torch.float32))
            K_inv_all, extrinsics_inv_all = torch.stack(K_inv_all), torch.stack(extrinsics_inv_all)

            eps = 1e-5
            #pad_h, pad_w = 84, 84
            B, C, H, W = img_feats.shape
            
            u, v = np.meshgrid(np.arange(pad_w), np.arange(pad_h))
            u, v = u.flatten(), v.flatten()
            px_coords = np.stack([u, v, np.ones_like(u)], axis=0)
            px_coords = torch.from_numpy(px_coords[:, None]).to(torch.float32).to(img_feats.device)

            depths=np.array([self.depth_start + (depth_index*self.depth_step) for depth_index in range(self.depth_num)])
            depths=torch.from_numpy(rearrange(depths, 'D -> 1 D 1')).to(torch.float32).to(img_feats.device)
            px_coords_all =  px_coords*depths

            # st=time.time()
            coords3ds = torch.einsum('mij,jkl->mikl', K_inv_all, px_coords_all)
            coords3ds = torch.cat([coords3ds, torch.ones_like(coords3ds[:,:1])], dim=1)
            coords3ds = torch.einsum('mij,mjkl->mikl', extrinsics_inv_all, coords3ds)
            coords3ds=rearrange(coords3ds[:, :3], 'C M D (H W) -> C H W (D M)', D=self.depth_num, H=pad_h, W=pad_w)
            coords3ds[..., 0::3] = (coords3ds[..., 0::3] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
            coords3ds[..., 1::3] = (coords3ds[..., 1::3] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
            coords3ds[..., 2::3] = (coords3ds[..., 2::3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
            # print(time.time()-st)

            if 0:
                cams = []
                for extrinsic, fov in zip(calibs['extrinsic_matrix'], calibs['fov']): 
                    cams.append(cam(np.array(extrinsic)*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]),pad_h,pad_w,fov))
                
                st=time.time()
                coords3ds_ = []
                for cam_ in cams:
                    coords3d = []
                    for depth_index in range(self.depth_num):
                        depth = self.depth_start + (depth_index*self.depth_step) 
                        coords3d_1d = deproject(cam_,depth,pad_h, pad_w).transpose(1, 0)
                        coords3d_1d = np.reshape(
                                                coords3d_1d, (pad_h, pad_w, 3)
                                            )
                        coords3d_1d[..., 0:1] = (coords3d_1d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
                        coords3d_1d[..., 1:2] = (coords3d_1d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
                        coords3d_1d[..., 2:3] = (coords3d_1d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
                        coords3d.append(coords3d_1d)
                    coords3d = np.concatenate(coords3d,axis=-1)
                    coords3ds_.append(coords3d)
                coords3ds_ = torch.from_numpy(np.array(coords3ds_)).to(img_feats.device)
                print(time.time()-st)
                print((coords3ds-coords3ds_).abs().max())

            if 0:
                st=time.time()
                coords3ds_ = []
                for cam_ in cams:
                    K = get_intrinsic_matrix(cam_)
                    K_inv = np.linalg.inv(K)
                    viewMatrix = cam_.viewMatrix * np.array([[1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1]])
                    extrinsics_inv = np.linalg.inv(viewMatrix)
                    K_inv_torch = torch.from_numpy(K_inv).to(img_feats.device).to(torch.float32)
                    extrinsics_inv_torch = torch.from_numpy(extrinsics_inv).to(img_feats.device).to(torch.float32)

                    px_coords_z_all = px_coords_all.to(img_feats.device)
                    # cam_coords_all = K_inv_torch @ px_coords_z_all
                    cam_coords_all = torch.einsum('ij,jkl->ikl', K_inv_torch, px_coords_z_all)
                    cam_coords_hom_all = torch.cat([cam_coords_all, torch.ones_like(cam_coords_all[:1])], dim=0)
                    # world_coords_all = extrinsics_inv_torch @ cam_coords_hom_all
                    world_coords_all = torch.einsum('ij,jkl->ikl', extrinsics_inv_torch, cam_coords_hom_all)
                    coords3d_1d_all=rearrange(world_coords_all[:3], 'M D (H W) -> H W (D M)', D=self.depth_num, H=pad_h, W=pad_w)
                    coords3d_1d_all[..., 0::3] = (coords3d_1d_all[..., 0::3] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
                    coords3d_1d_all[..., 1::3] = (coords3d_1d_all[..., 1::3] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
                    coords3d_1d_all[..., 2::3] = (coords3d_1d_all[..., 2::3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
                    coords3d=coords3d_1d_all
                coords3ds_.append(coords3d)
                coords3ds_ = torch.stack(coords3ds_)
                print(time.time()-st)

            coords3ds = coords3ds.permute(0, 3, 1, 2)
            coords3ds = F.interpolate(
                    coords3ds,
                    (H, W),
                    mode='bilinear'
                )
            coords3ds = inverse_sigmoid(coords3ds.float())
        coords3ds_clone = coords3ds.clone().detach()
        coords_position_embeding = self.position_encoder(coords3ds_clone.to(img_feats.dtype))
        del coords3ds, px_coords_all, px_coords, depths
        return coords_position_embeding#.view(B, N, self.embed_dims, H, W)
    

   
   
    def forward(self, mlvl_feats, calibs, pos, pad_h=84, pad_w=84,fov=0):
        """Forward function.
       
        """
        TESTTIME=False; st_yf = time.time()
        assert fov==0 # 注意此处fov未使用，因为不同的数据fov不同

        x = mlvl_feats
        batch_size = x.size(0)
        coords_position_embeding = self.position_embeding(mlvl_feats, calibs, pad_h, pad_w)
        pos_embed = coords_position_embeding
        # pos_embeds = []
        # for i in range(num_cams):
        if TESTTIME: ed_pe_yf=time.time(); print(f"YF: position_embeding {ed_pe_yf-st_yf}")

        sin_embed = pos
        sin_embed = self.adapt_pos3d(sin_embed)
        pos_embed = pos_embed + sin_embed

        if TESTTIME: ed_cn_yf=time.time(); print(f"YF: adapt_pos3d {ed_cn_yf-ed_pe_yf}")
        return pos_embed

    
    




def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)



def deproject(cam, depth, pad_h, pad_w,homogeneous=False, sanity_check=False):
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
    depth_img = np.ones((pad_h, pad_w))*depth
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix))
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
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

def get_intrinsic_matrix(cam):
    """Construct the intrinsic matrix from camera parameters."""
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    K = np.array([
        [foc, 0, cam.width // 2],
        [0, foc, cam.height // 2],
        [0, 0, 1]
    ])
    return K

def convert_to_world_coordinates(cam, depth, pad_h, pad_w):
    """Convert pixel coordinates to world coordinates using intrinsic and extrinsic inverses."""
    u, v = np.meshgrid(np.arange(pad_w), np.arange(pad_h))
    u, v = u.flatten(), v.flatten()
    
    # Compute the depth for each pixel
    z = np.ones((pad_h, pad_w)) * depth
    z = z.flatten()

    # Create the intrinsic matrix and its inverse
    K = get_intrinsic_matrix(cam)
    K_inv = np.linalg.inv(K)

    # Stack the coordinates to form homogeneous pixel coordinates (3, N)
    px_coords = np.stack([u, v, np.ones_like(u)], axis=0)

    # Project pixel coordinates back to camera coordinates
    cam_coords = np.dot(K_inv, px_coords) * z

    # Transform camera coordinates to world coordinates
    viewMatrix = cam.viewMatrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    T_world_cam = np.linalg.inv(np.array(viewMatrix))
    cam_coords_hom = np.vstack([cam_coords, np.ones_like(z)])  # Homogeneous coordinates
    world_coords = np.dot(T_world_cam, cam_coords_hom)
    
    return world_coords[:3]  # Return only the x, y, z coordinates


def convert_to_world_coordinates(cam, depth, pad_h, pad_w):

    u, v = np.meshgrid(np.arange(pad_w), np.arange(pad_h))
    u, v = u.flatten(), v.flatten()
    
    # Compute the depth for each pixel
    z = np.ones((pad_h, pad_w)) * depth
    z = z.flatten()
    px_coords = np.stack([u, v, np.ones_like(u)], axis=0)* z


    intrinsics = get_intrinsic_matrix(cam)
    extrinsics = cam.viewMatrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]

    print(cam_proj_mat_inv.shape, px_coords.shape, np.ones_like(u)[None].shape, np.vstack([px_coords, np.ones_like(u)[None]]).shape)

    world_coords = np.dot(cam_proj_mat_inv, np.vstack([px_coords, np.ones_like(u)[None]])) 

    return world_coords

def get_intrinsic_matrix(cam):
    """Construct the intrinsic matrix from camera parameters."""
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    K = np.array([
        [foc, 0, cam.width // 2],
        [0, foc, cam.height // 2],
        [0, 0, 1]
    ])
    return K


# # Create the intrinsic matrix and its inverse
# K = get_intrinsic_matrix(cam)
# K_inv = np.linalg.inv(K)
# # Adjust for coordinate system differences
# viewMatrix = cam.viewMatrix * np.array([[1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1]])

# # Compute the inverse of the camera extrinsic matrix
# extrinsics_inv = np.linalg.inv(viewMatrix)

def convert_to_world_coordinates(K_inv, extrinsics_inv, depth, pad_h, pad_w):
    """Convert pixel coordinates to world coordinates using intrinsic and extrinsic inverses."""
    u, v = np.meshgrid(np.arange(pad_w), np.arange(pad_h))
    u, v = u.flatten(), v.flatten()

    # Compute the depth for each pixel
    z = np.ones_like(u) * depth

    # Stack the coordinates to form homogeneous pixel coordinates (3, N)
    px_coords = np.stack([u, v, np.ones_like(u)], axis=0)

    # Project pixel coordinates back to camera coordinates
    cam_coords = np.dot(K_inv, px_coords) * z

    # Apply extrinsics inverse to get world coordinates
    cam_coords_hom = np.vstack([cam_coords, np.ones_like(z)])  # Homogeneous coordinates
    world_coords = np.dot(extrinsics_inv, cam_coords_hom)

    return world_coords[:3]  # Return only the x, y, z coordinates

# from pyrep.objects import VisionSensor
# r_sh_depth_m = np.ones((pad_h, pad_w)) * depth
# extrinsics=np.linalg.inv(cam.viewMatrix) * np.array([[1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1]])
# intrinsics = get_intrinsic_matrix(cam)
# VisionSensor.pointcloud_from_depth_and_camera_params(r_sh_depth_m, extrinsics, intrinsics)




        #     for depth_index in range(self.depth_num):
        #         depth = self.depth_start + (depth_index*self.depth_step) 
        #         # coords3d_1d = convert_to_world_coordinates(K_inv, extrinsics_inv,depth,pad_h, pad_w).transpose(1, 0)

        #         px_coords_z = torch.from_numpy(px_coords*np.ones_like(u) * depth).to(img_feats.device).to(torch.float32)
        #         # Project pixel coordinates back to camera coordinates
        #         cam_coords = K_inv_torch @ px_coords_z
        #         # Apply extrinsics inverse to get world coordinates
        #         cam_coords_hom = torch.cat([cam_coords, torch.ones_like(cam_coords[:1])], dim=0)
        #         world_coords = extrinsics_inv_torch @ cam_coords_hom
        #         coords3d_1d=world_coords[:3].transpose(1,0)

        #         coords3d_1d = coords3d_1d.reshape(pad_h, pad_w, 3)
        #         coords3d_1d[..., 0:1] = (coords3d_1d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        #         coords3d_1d[..., 1:2] = (coords3d_1d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        #         coords3d_1d[..., 2:3] = (coords3d_1d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
        #         coords3d.append(coords3d_1d)

        #     coords3d = torch.cat(coords3d, axis=-1)
        #     coords3ds.append(coords3d)
        # coords3ds = torch.stack(coords3ds)
        # print(time.time()-st)
