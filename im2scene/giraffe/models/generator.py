import torch.nn as nn
import torch.nn.functional as F
import torch
from im2scene.common import (
    arange_pixels, image_points_to_world, origin_to_world
)
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from im2scene.camera import get_camera_mat, get_random_pose, get_camera_pose
import pdb 
from .resnet import resnet34

class Generator(nn.Module):
    ''' GIRAFFE Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        z_dim_bg (int): dimension of background latent code z_bg
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        background_generator (nn.Module): background generator
        bounding_box_generaor (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
        sample_object-existance (bool): whether to sample the existance
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    '''

    def __init__(self, device, z_dim=256, z_dim_bg=128, decoder=None,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 background_generator=None,
                 bounding_box_generator=None, resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 backround_rotation_range=[0., 0.],
                 sample_object_existance=False,
                 use_max_composition=False, **kwargs):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.bounding_box_generator = bounding_box_generator
        self.fov = fov
        self.backround_rotation_range = backround_rotation_range
        self.sample_object_existance = sample_object_existance
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.use_max_composition = use_max_composition
        self.resnet = resnet34(pretrained=True, shape_dim=self.z_dim, app_dim=self.z_dim)
        self.camera_matrix = get_camera_mat(fov=fov).to(device) # intrinsic camera 
        self.range_v = range_v
        self.range_u = range_u

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if background_generator is not None:
            self.background_generator = background_generator.to(device)
        else:
            self.background_generator = None
        if bounding_box_generator is not None:
            self.bounding_box_generator = bounding_box_generator.to(device)
        else:
            self.bounding_box_generator = bounding_box_generator
        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self, img, pose, batch_size=32, latent_codes=None, camera_matrices=None,
                transformations=None, bg_rotation=None, mode="training", it=0,
                return_alpha_map=False,
                not_render_background=False,
                only_render_background=False,
                need_uv=False):
        # edit mira start 
        # 랜덤하게 두개만 뽑자     
        img1, img2 = img[0].to(self.device), img[1].to(self.device)
        pose1, pose2 = pose[:, 0], pose[:, 1]

        batch_size = img1.shape[0]
        # uv, shape, appearance = self.resnet(img)    # img 넣어주기 (Discriminator에서 input으로 받는거 그대로)
        img = torch.cat((img1, img2), dim=0)        
        shape, appearance = self.resnet(img)    # img 넣어주기 (Discriminator에서 input으로 받는거 그대로)
        # 만약 gradient 저장이 문제라면 

        mid = int(len(shape) / 2)

        if latent_codes is None:
            latent_codes1 = shape[:mid].unsqueeze(1), appearance[:mid].unsqueeze(1)       # OK
            latent_codes2 = shape[mid:].unsqueeze(1), appearance[mid:].unsqueeze(1)       # OK


        if camera_matrices is None: 
            # camera, world matrices 
            # camera matrices 어떻게 고정하지? 
            random_u = torch.rand(batch_size)*(self.range_u[1] - self.range_u[0]) + self.range_u[0]
            random_v = torch.rand(batch_size)*(self.range_v[1] - self.range_v[0]) + self.range_v[0]
            # 정해진 batch size만큼 나옴 
            rand_camera_matrices = self.get_random_camera(random_u, random_v, batch_size)
            intrinsic_mat = rand_camera_matrices[0]
            # u, v = uv[:, 0], uv[:, 1]
            # v = v/2

            camera_matrices = tuple((intrinsic_mat, pose1))    # 앞부분 mat 
            new_cam_matrices = tuple((intrinsic_mat, pose2))   # 뒷부분 mat 
            swap_camera_matrices = tuple((intrinsic_mat, pose1.flip(0)))    # 앞부분 mat 


        else:
            print('oh noooo')
            pdb.set_trace()

        if transformations is None:
            # transformations = pose
            # swap_transformations = pose.flip(0)
            transformations = self.get_random_transformations(batch_size)     

        # edit mira end 

        if return_alpha_map:
            rgb_v, alpha_map = self.volume_render_image(
                latent_codes1, camera_matrices, 
                mode=mode, it=it, return_alpha_map=True,
                not_render_background=not_render_background)
            return alpha_map
        else:
            rgb_v = self.volume_render_image(
                latent_codes1, camera_matrices, transformations, 
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background)
            rgb_v2 = self.volume_render_image(
                latent_codes2, new_cam_matrices, transformations, 
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background)
            swap_rgb_v = self.volume_render_image(
                latent_codes1, swap_camera_matrices, transformations, 
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background)
            # rand_rgb_v = self.volume_render_image(
            #     latent_codes1, rand_camera_matrices, transformations, 
            #     mode=mode, it=it, not_render_background=not_render_background,
            #     only_render_background=only_render_background)


            if self.neural_renderer is not None:
                rgb = self.neural_renderer(rgb_v)
                rgb2 = self.neural_renderer(rgb_v2)
                swap_rgb = self.neural_renderer(swap_rgb_v)
                # rand_rgb = self.neural_renderer(rand_rgb_v)
            else:
                rgb = rgb_v
                rgb2 = rgb_v2
                swap_rgb = swap_rgb_v
                # rand_rgb = rand_rgb_v

            if need_uv==True:
                # return rgb, rgb2, swap_rgb, rand_rgb, torch.cat((random_u.unsqueeze(-1), random_v.unsqueeze(-1)), dim=-1)
                return rgb, rgb2, swap_rgb, torch.cat((random_u.unsqueeze(-1), random_v.unsqueeze(-1)), dim=-1)

                # return rgb, swap_rgb, rand_rgb, (uv, uv.flip(0), torch.cat((random_u.unsqueeze(-1), random_v.unsqueeze(-1)), dim=-1))
                # return rgb, swap_rgb, rand_rgb, torch.cat((rand_camera_matrices[1], pose, pose.flip(0)), dim=-1)

            else:
                return rgb, rgb2, swap_rgb

    def get_n_boxes(self):
        if self.bounding_box_generator is not None:
            n_boxes = self.bounding_box_generator.n_boxes
        else:
            n_boxes = 1
        return n_boxes

    def get_latent_codes(self, batch_size=32, tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg
        n_boxes = self.get_n_boxes()
        def sample_z(x): return self.sample_z(x, tmp=tmp)
        z_shape_obj = sample_z((batch_size, n_boxes, z_dim))
        z_app_obj = sample_z((batch_size, n_boxes, z_dim))
        z_shape_bg = sample_z((batch_size, z_dim_bg))
        z_app_bg = sample_z((batch_size, z_dim_bg))
        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg     # z_shape_obj = (16, 1, 256), n_boxes = 1

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z

    def get_vis_dict(self, batch_size=32):
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(-1, -1, batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        return vis_dict


    def get_random_camera(self, u=None, v=None, batch_size=32, to_device=True): 
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)    # intrinsic camera
        world_mat = get_random_pose(
            u, v, self.range_radius, batch_size)      # (batch, 4, 4)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32,
                   to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_camera_pose(
            self.range_u, self.range_v, self.range_radius, val_u, val_v,
            val_r, batch_size=batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_random_bg_rotation(self, batch_size, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(Rot.from_euler(
                    'z', r_random * 2 * np.pi).as_dcm()
                ) for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            R_bg = R_bg.to(self.device)
        return R_bg

    def get_bg_rotation(self, val, batch_size=32, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_dcm()
            ).reshape(1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            r = r.to(self.device)
        return r

    def get_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations(self, val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]], val_r=[0.5],
                            batch_size=32, to_device=True):
        device = self.device
        s = self.bounding_box_generator.get_scale(
            batch_size=batch_size, val=val_s)
        t = self.bounding_box_generator.get_translation(
            batch_size=batch_size, val=val_t)
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)

        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations_in_range(self, range_s=[0., 1.], range_t=[0., 1.],
                                     range_r=[0., 1.], n_boxes=1,
                                     batch_size=32, to_device=True):
        s, t, R = [], [], []

        def rand_s(): return range_s[0] + \
            np.random.rand() * (range_s[1] - range_s[0])

        def rand_t(): return range_t[0] + \
            np.random.rand() * (range_t[1] - range_t[0])
        def rand_r(): return range_r[0] + \
            np.random.rand() * (range_r[1] - range_r[0])

        for i in range(batch_size):
            val_s = [[rand_s(), rand_s(), rand_s()] for j in range(n_boxes)]
            val_t = [[rand_t(), rand_t(), rand_t()] for j in range(n_boxes)]
            val_r = [rand_r() for j in range(n_boxes)]
            si, ti, Ri = self.get_transformations(
                val_s, val_t, val_r, batch_size=1, to_device=to_device)
            s.append(si)
            t.append(ti)
            R.append(Ri)
        s, t, R = torch.cat(s), torch.cat(t), torch.cat(R)
        if to_device:
            device = self.device
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_rotation(self, val_r, batch_size=32, to_device=True):
        device = self.device
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)

        if to_device:
            R = R.to(device)
        return R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0,    # p = (batch, 256, 3) = (batch, 16x16, 3)
                                scale_factor=1.):
        # edit mira start 
        # 여기는 조금 공부를 해야할듯.. 
        batch_size = p.shape[0]
        # if not isinstance(transformations, tuple):      # len: 3
        #     tensor_device = transformations.device
        #     bb_s = torch.tensor([1, 1, 1]).reshape(1, 1, 3).repeat(batch_size, 1, 1).to(tensor_device)        # t = 그냥 t or R-1의 t..?
            
        #     # translation 그냥 넣기 
        #     bb_t = transformations[:, -1, :3].reshape(-1, 1, 3)

        #     bb_R = transformations[:, :3, :3].reshape(-1, 1, 3, 3)
        # else:
        # bb_s, bb_t, bb_R = transformations

        # bb_s, bb_t, bb_R = transformations      # s: (batch, 1, 3), t: (batch, 1, 3), R: (batch, 1, 3, 3)
        # p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)
        #                              ).permute(0, 2, 1)).permute(
        #     0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        # edit mira end
        return p

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        camera_world = (rotation_matrix @
                        camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (rotation_matrix @
                        pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world

        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations)
        ray_i = pixels_world_i - camera_world_i

        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def get_object_existance(self, n_boxes, batch_size=32):
        '''
        Note: We only use this setting for Clevr2345, so that we can hard-code
        the probabilties here. If you want to apply it to a different scenario,
        you would need to change these.
        '''
        probs = [
            .19456788355146545395,
            .24355003312266127155,
            .25269546846185522711,
            .30918661486401804737,
        ]

        n_objects_prob = np.random.rand(batch_size)
        n_objects = np.zeros_like(n_objects_prob).astype(np.int)
        p_cum = 0
        obj_n = [i for i in range(2, n_boxes + 1)]
        for idx_p in range(len(probs)):
            n_objects[
                (n_objects_prob >= p_cum) &
                (n_objects_prob < p_cum + probs[idx_p])
            ] = obj_n[idx_p]
            p_cum = p_cum + probs[idx_p]
            assert(p_cum <= 1.)

        object_existance = np.zeros((batch_size, n_boxes))
        for b_idx in range(batch_size):
            n_obj = n_objects[b_idx]
            if n_obj > 0:
                idx_true = np.random.choice(
                    n_boxes, size=(n_obj,), replace=False)
                object_existance[b_idx, idx_true] = True
        object_existance = object_existance.astype(np.bool)
        return object_existance

    def volume_render_image(self, latent_codes, camera_matrices,
                            transformations, mode='training',       # transformation: 의미 X 
                            it=0, return_alpha_map=False,
                            not_render_background=False,
                            only_render_background=False):
        res = self.resolution_vol           # 16 
        device = self.device    
        n_steps = self.n_ray_samples        # 64
        n_points = res * res                # 16x16 = 256
        depth_range = self.depth_range      # [0.5, 6.0]
        batch_size = latent_codes[0].shape[0]   
        z_shape_obj, z_app_obj = latent_codes     # (32,1,256), (32,1,256), [(32,128), (32,128) -> delete!]
        assert(not (not_render_background and only_render_background))

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size,      # imgrange (-1., 1.) -> scaled pixel from -1 to 1만 가져옴! -> meshgrid!  Arranges pixels for given resolution in range image_range.
                               invert_y_axis=False)[1].to(device)   # (batch, 16(res)x16(res), 2(xy))
        pixels[..., -1] *= -1.  # y축에 -1을 곱함. -> scale 맞춰주려구! -1이어야 image plane보다 뻗어나가나봄!
        # Project to 3D world
        pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],     
            world_mat=camera_matrices[1])       # fixed depth value 1까지 고려해서 -> (batch, resxres, 3)       # worldmat: randomly sampled
        camera_world = origin_to_world(     
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])        # (1, 1, 64)
        di = di.repeat(batch_size, n_points, 1).to(device)  # (batch, n_points, 64)
        if mode == 'training':
            di = self.add_noise_to_interval(di)     # 균일하게 뽑지 않고 noise를 더해서 뽑음!

        n_boxes = latent_codes[0].shape[1]      # object가 하나라 1로 예측중!
        feat, sigma = [], []
        p_i, r_i = self.get_evaluation_points(      # points들을 transforation을 사용해서 가져옴
            pixels_world, camera_world, di, transformations)
        z_shape_i, z_app_i = z_shape_obj[:, 0], z_app_obj[:, 0]

        # 여기 이하를 전부 가져오기!
        # 여기 self.decoder 부분을 가져오기!
        feat_i, sigma_i = self.decoder(p_i, r_i, z_shape_i, z_app_i)

        if mode == 'training':
            # As done in NeRF, add noise during training
            sigma_i += torch.randn_like(sigma_i)

        # Mask out values outside
        padd = 0.1
        mask_box = torch.all(
            p_i <= 1. + padd, dim=-1) & torch.all(
                p_i >= -1. - padd, dim=-1)
        sigma_i[mask_box == 0] = 0.

        # Reshape
        sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
        feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)

        feat.append(feat_i)
        sigma.append(sigma_i)
        sigma = F.relu(torch.stack(sigma, dim=0))
        feat = torch.stack(feat, dim=0)
        # edit mira end 
        if self.sample_object_existance:
            # 얘 false임!!! -> PASS!
            object_existance = self.get_object_existance(n_boxes, batch_size)
            # add ones for bg
            object_existance = np.concatenate(
                [object_existance, np.ones_like(
                    object_existance[..., :1])], axis=-1)
            object_existance = object_existance.transpose(1, 0)
            sigma_shape = sigma.shape
            sigma = sigma.reshape(sigma_shape[0] * sigma_shape[1], -1)
            object_existance = torch.from_numpy(object_existance).reshape(-1)
            # set alpha to 0 for respective objects
            sigma[object_existance == 0] = 0.
            sigma = sigma.reshape(*sigma_shape)


        # Composite
        sigma_sum, feat_weighted = self.composite_function(sigma, feat)

        # Get Volume Weights <- calc_volume_weights를 하기 위한 di, ray_vector, sigma_sum은 어디에..? 
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        if return_alpha_map:
            n_maps = sigma.shape[0]
            acc_maps = []
            for i in range(n_maps - 1):
                sigma_obj_sum = torch.sum(sigma[i:i+1], dim=0)
                weights_obj = self.calc_volume_weights(
                    di, ray_vector, sigma_obj_sum, last_dist=0.)
                acc_map = torch.sum(weights_obj, dim=-1, keepdim=True)
                acc_map = acc_map.permute(0, 2, 1).reshape(
                    batch_size, -1, res, res)
                acc_map = acc_map.permute(0, 1, 3, 2)
                acc_maps.append(acc_map)
            acc_map = torch.cat(acc_maps, dim=1)
            return feat_map, acc_map
        else:
            return feat_map
