from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average, make_patchgan_target)
from torchvision.utils import save_image, make_grid
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np 
import pdb 
import math 
import torch.nn.functional as F
from torch import nn


logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 val_vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, batch_size=None, recon_weight=None, cam_loss_weight=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.val_vis_dir = val_vis_dir
        self.multi_gpu = multi_gpu

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations
        self.recon_loss = torch.nn.MSELoss()
        self.recon_weight = recon_weight
        self.cam_loss_weight = cam_loss_weight

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        self.patch_discriminator = self.model.patch_discriminator

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if val_vis_dir is not None and not os.path.exists(val_vis_dir):
            os.makedirs(val_vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        loss_gen, deterministic_loss, gloss_fake, gloss_mix, gloss_swapcam, ploss_mix, cam_GT_loss, recon_loss = self.train_step_generator(data, it)
        loss_d, reg_d, real_d, mix_d, swap_d, fake_d, real_p, mix_p = self.train_step_discriminator(data, it)

        return {
            'gen': loss_gen,
            'det_loss': deterministic_loss, 
            'fake_g': gloss_fake, 
            'mix_g': gloss_mix, 
            'swapcam_g': gloss_swapcam, 
            'mix_p': ploss_mix,
            'cam_GT_loss': cam_GT_loss,
            'recon_loss': recon_loss,

            'disc': loss_d,
            'regularizer': reg_d,
            'real_d': real_d,
            'mix_d': mix_d,
            'swap_d': swap_d,
            'fake_d': fake_d,
            'real_p': real_p,
            'mix_p': mix_p,
        }

        # loss_gen, fake_g = self.train_step_generator(data, it)
        # loss_d, reg_d, real_d, fake_d = self.train_step_discriminator(data, it)

        # return {
        #     'gen_total': loss_gen,
        #     'fake_g': fake_g,
        #     'disc_total': loss_d,
        #     'regularizer': reg_d,
        #     'real_d': real_d,
        #     'fake_d': fake_d,
        # }


    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake, _, _, _ = generator(x_real, pose_real)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        patchgan_loss = nn.BCEWithLogitsLoss()
        try:
            _ = discriminator.in_dim
            discriminator_type = 'plainD'
        except:
            discriminator_type = 'patchD'

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)

        # 얘네 왜함..?
        generator.train()
        #generator.ViT.eval()
        discriminator.train()

        self.optimizer.zero_grad()

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)
        # mask_real = data.get('mask').to(self.device)
        

        # find gt_trans
        #gt_trans = pose_real[:, :3, -1]
        # proj w/o trans into scale, rotation은 여기서는 필요없기 때문에 패스! -> numpy, pytorch를 왔다갔다해야하기 때문에 시간 많이 잡아먹음 
        gt_rot = pose_real[:, :3, :3]
        #gt_scale = torch.tensor([1.]).reshape(-1, 1).repeat(len(gt_rot), 1).to(self.device)
        
        if self.multi_gpu:
            latents = generator.module.get_vis_dict(x_real.shape[0])
            x_fake, x_mix, pred_cam, x_swapcam = generator(x_real, pose_real, **latents)       

        else:
            x_fake, x_mix, pred_cam, x_swapcam = generator(x_real, pose_real)  #x_mix -> shape swapped

        
        # cam pred loss 주기 
        rot_from_GT = pred_cam[:, :3, :3]
        
        cam_GT_loss = torch.norm(torch.bmm(torch.inverse(rot_from_GT), gt_rot) - torch.eye(3).reshape(-1, 3, 3).repeat(len(gt_rot), 1, 1).to(self.device))
        #import pdb; pdb.set_trace()
        cam_GT_loss = cam_GT_loss + torch.norm(rot_from_GT[:, :3, 2] - gt_rot[:, :3, 2])
        #cam_GT_loss = torch.norm(rot_from_GT[:, :3, 2] - gt_rot[:, :3, 2])
        #import pdb; pdb.set_trace()
        #cam_GT_loss = torch.zeros(1).cuda()

        recon_loss = self.recon_loss(x_fake, x_real) * self.recon_weight
        deterministic_loss = cam_GT_loss*self.cam_loss_weight + recon_loss
        '''cam_GT_loss = torch.zeros(1).to(self.device)
        recon_loss = torch.zeros(1).to(self.device)
        deterministic_loss = torch.zeros(1).to(self.device)'''
        
        # insert gan 
        d_fake = discriminator(x_fake)
        d_mix = discriminator(x_mix)
        d_swapcam = discriminator(x_swapcam)
        #import pdb; pdb.set_trace()

        if discriminator_type == 'patchD':
            d_fake_target = make_patchgan_target(d_fake, need_true = True)
            gloss_fake = patchgan_loss(d_fake, d_fake_target)
            d_mix_target = make_patchgan_target(d_mix, need_true = True)
            gloss_mix = patchgan_loss(d_mix, d_mix_target)
            d_swapcam_target = make_patchgan_target(d_swapcam, need_true = True)
            gloss_swapcam = patchgan_loss(d_swapcam, d_swapcam_target)

        else:
            gloss_fake = compute_bce(d_fake, 1)
            gloss_mix = compute_bce(d_mix, 1)
            gloss_swapcam = compute_bce(d_swapcam, 1) #patchgan loss로
        #gloss_fake = torch.zeros(1).cuda()
        #gloss_mix = torch.zeros(1).cuda()
        #gloss_swapcam = torch.zeros(1).cuda()
        # gloss_mix = self.gan_loss(self.PattGAN(real_img, fake_img), should_be_classified).mean()



        # for patch co-occurrence discriminator 
        if self.patch_discriminator is not None:
            real_feat = self.patch_discriminator.extract_features(
                self.get_random_crops(x_real),
                aggregate=self.patch_discriminator.patch_use_aggregation).detach()
            mix_feat = self.patch_discriminator.extract_features(self.get_random_crops(x_mix))

            ploss_mix = self.gan_loss(
                    self.patch_discriminator.discriminate_features(real_feat, mix_feat),
                    should_be_classified_as_real=True,
                ).mean()
            gen_loss = deterministic_loss + (gloss_fake + gloss_mix + gloss_swapcam)/3 + ploss_mix * self.patch_discriminator.lambda_PatchGAN
        else:
            ploss_mix = torch.zeros(1)
            gen_loss = deterministic_loss + (gloss_fake + gloss_mix + gloss_swapcam)/3
        
 
        gen_loss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gen_loss.item(), deterministic_loss.item(), gloss_fake.item(), gloss_mix.item(), gloss_swapcam.item(), \
                                        ploss_mix.item(), cam_GT_loss.item(), recon_loss.item()


    def gan_loss(self, pred, should_be_classified_as_real):
        bs = pred.size(0)
        if should_be_classified_as_real:
            return F.softplus(-pred).view(bs, -1).mean(dim=1)       # 이 자체가 activation output으로서 기능 
        else:
            return F.softplus(pred).view(bs, -1).mean(dim=1)

    ''' from swapping autoencoder start'''
    #### geneator loss 
    def compute_generator_losses(self, real, sp_ma, gl_ma):
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        
        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec, real[:B // 2])

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            real = real[B // 2:]
            gl = gl[B // 2:]
            sp_mix = sp_mix[B // 2:]

        mix = self.G(sp_mix, gl)

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                self.D(mix),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.patch_discriminator.extract_features(
                self.get_random_crops(real),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.patch_discriminator.extract_features(self.get_random_crops(mix))

            losses["G_mix"] = loss.gan_loss(
                self.patch_discriminator.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN

        return losses, metrics


    #### discriminator loss 
    def compute_discriminator_losses(self, real):
        self.num_discriminator_iters.add_(1)

        sp, gl = self.E(real)
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images              # 굳이....?
        rec = self.G(sp[:B // 2], gl[:B // 2])      # sp: spatial, gl: global
        mix = self.G(self.swap(sp), gl)

        losses = self.compute_image_discriminator_losses(real, rec, mix)        # 그냥 gan loss 

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.patch_discriminator.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.patch_discriminator.extract_features(target_crop)
            pred_real_patch = self.patch_discriminator.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    ''' from swapping autoencoder end'''


    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        patchgan_loss = nn.BCEWithLogitsLoss()
        try:
            _ = discriminator.in_dim
            discriminator_type = 'plainD'
        except:
            discriminator_type = 'patchD'

        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        #generator.ViT.eval()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)
        if discriminator_type == 'patchD':
            d_swapcam_target = make_patchgan_target(d_real, need_true = True)
            d_loss_real = patchgan_loss(d_real, d_swapcam_target)
        else:
            d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict(x_real.shape[0])
                x_fake, x_mix, _, x_swapcam = generator(x_real, pose_real, **latents)        

            else:
                x_fake, x_mix, _, x_swapcam = generator(x_real, pose_real)

        # r1 regularization 어떻게 들어가는지 다시 한번 봐보기 
        x_mix.requires_grad_()
        x_fake.requires_grad_()
        x_swapcam.requires_grad_()
        d_mix = discriminator(x_mix)
        d_fake = discriminator(x_fake)
        d_swapcam = discriminator(x_swapcam)
        # for whole image discriminator 
        if discriminator_type == 'patchD':
            d_mix_target = make_patchgan_target(d_mix, need_true = False)
            d_loss_mix = patchgan_loss(d_mix, d_mix_target)
            loss_d_full += d_loss_mix / 3
            
            d_fake_target = make_patchgan_target(d_fake, need_true = False)
            d_loss_fake = patchgan_loss(d_fake, d_fake_target)
            loss_d_full += d_loss_fake / 3

            d_swapcam_target = make_patchgan_target(d_swapcam, need_true = False)
            d_loss_swapcam = patchgan_loss(d_swapcam, d_swapcam_target)
            loss_d_full += d_loss_swapcam / 3

        else:
            d_loss_fake = compute_bce(d_fake, 0)
            d_loss_mix = compute_bce(d_mix, 0)
            d_loss_swapcam = compute_bce(d_swapcam, 0)
            #d_loss_mix = torch.zeros(1).cuda()
            #d_loss_swapcam = torch.zeros(1).cuda()
            loss_d_full = loss_d_full + (d_loss_fake + d_loss_mix + d_loss_swapcam)/3

        # for patch co-occurrence discriminator 
        if self.patch_discriminator is not None:
            real_feat = self.patch_discriminator.extract_features(self.get_random_crops(x_real), aggregate=True)    # 뭐지 얘는 집합인건가 
            target_feat = self.patch_discriminator.extract_features(self.get_random_crops(x_real))
            mix_feat = self.patch_discriminator.extract_features(self.get_random_crops(x_mix))

            ploss_real = self.gan_loss(self.patch_discriminator.discriminate_features(real_feat, target_feat), should_be_classified_as_real=True).mean()
            ploss_mix = self.gan_loss(self.patch_discriminator.discriminate_features(real_feat, mix_feat), should_be_classified_as_real=False).mean()
            loss_d_full += (ploss_real + ploss_mix) * self.patch_discriminator.lambda_PatchGAN 
        else:
            ploss_real = torch.zeros(1)
            ploss_mix = torch.zeros(1)
        # value들의 평균을 더해서 sum한다   # sum(v.mean()) for each loss category 
        #loss_d_full += (ploss_real + ploss_mix) * self.patch_discriminator.lambda_PatchGAN   # 얘의 weight 따로 있음 
 
        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_real + (d_loss_fake + d_loss_mix + d_loss_swapcam) / 3) 

        return (
            d_loss.item(), reg.item(), d_loss_real.item(), d_loss_mix.item(), d_loss_swapcam.item(), d_loss_fake.item(), ploss_real, ploss_mix)


    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = self.apply_random_crop(
            x, self.patch_discriminator.patch_size,
            (self.patch_discriminator.patch_min_scale, self.patch_discriminator.patch_max_scale),
            num_crops=self.patch_discriminator.patch_num_crops
        )
        return crops


    def apply_random_crop(self, x, target_size, scale_range, num_crops=1, return_rect=False):
        # build grid
        B = x.size(0) * num_crops
        flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
        unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
        unit_grid_y = unit_grid_x.transpose(1, 2)
        unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)


        #crops = []
        x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
        #for i in range(num_crops):
        scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
        sampling_grid = unit_grid * scale + offset
        crop = F.grid_sample(x, sampling_grid, align_corners=False)
        #crops.append(crop)
        #crop = torch.stack(crops, dim=1)
        crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

        return crop

    def record_uvs(self, uv, path, it):
        out_path = os.path.join(path, 'uv.txt')
        name_dict = {0: 'GT', 1: 'swap_GT', 2:'rand'}
        if not os.path.exists(out_path):
            f = open(out_path, 'w')
        else:
            f = open(out_path, 'a')

        uvs = torch.stack(uv, dim=0)
        for i in range(len(uvs)):        # len: 3
            line = list(map(lambda x: round(x, 3), uvs[i].flatten().detach().cpu().numpy().tolist()))
            out = []
            for idx in range(0, len(line)//2):
                out.append(tuple((line[2*idx], line[2*idx+1])))
            txt_line = f'{it}th {name_dict[i]}-uv : {out}\n'
            f.write(txt_line)
        f.write('\n')
        f.close()

    def uv2rad(self, uv):
        theta = 360 * uv[:, 0]
        phi = torch.arccos(1 - 2 * uv[:, 1]) / math.pi * 180
        
        return torch.stack([theta, phi], dim=-1) #16, 2

    def loc2rad(self, loc):
        phi = torch.acos(loc[:, 2])
        plusidx = torch.where(loc[:,1]<0)[0]
        theta = torch.acos(loc[:,0]/torch.sin(phi))
        theta[plusidx] = 2*math.pi-theta[plusidx]
        theta, phi = theta * 180 / torch.pi, phi * 180 / torch.pi
        return torch.cat([theta, phi], dim=-1)


    def visualize(self, data, it=0, mode=None, val_idx=None):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''


        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            # edit mira start 
            x_real = data.get('image').to(self.device)
            x_pose = data.get('pose').to(self.device)
            # image_fake, image_fake2, image_swap, uvs = self.generator(x_real, x_pose, mode='val', need_uv=True)
            # image_fake, image_fake2, image_swap = image_fake.detach(), image_fake2.detach(), image_swap.detach()
            image_fake, shape_rgb, appearance_rgb, swap_rgb = self.generator(x_real, x_pose, mode='val', need_uv=True)
            image_fake, shape_rgb, appearance_rgb, swap_rgb = image_fake.detach(), shape_rgb.detach(), appearance_rgb.detach(), swap_rgb.detach()

            # edit mira end 


            # import pdb 
            # pdb.set_trace()
            # xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[     # xyz를 self.poses로 rotate -> transpose했던거에 다시 연산됨..! -> 헐 그러면 이 상태에서 예측하는건가보다 그러면 이게 sampled points고 앞에 camera가 원래 ray에서...!
            #                                                                             # 그래야 transformation이 말이 되는 듯.. 근데 그럴거면 왜 transformation을 예측하지..? 
            #     ..., 0                                                                  # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!)
            # ]
            # # 오키.. def encoder에서 생긴 얘가 여기로 들어감!
            # xyz = xyz_rot + self.poses[:, None, :3, 3]      # 얘네가 sampling points!     # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!) 


            '''
            # 여기 안을 잘 조정하면 -> swapped view에서도 비슷한 맥락으로 나올 듯 
            # edit 'ㅅ'
            randrad = self.uv2rad(uvs).cuda()        # uv를 radian으로 표현 
            rotmat1 = x_pose[:,:3,:3]        # x_pose : real pose that includes R,t     # rotation matrix 
            # rotmat2 = x_pose[:, 1][:,:3,:3]        # x_pose : real pose that includes R,t
            # import pdb 
            # pdb.set_trace()
            origin = torch.Tensor([0,0,1]).to(self.device).repeat(int(len(x_pose)),1).unsqueeze(-1)
            '''
            # translation을 먼저 함 
            # camloc1 = rotmat1@origin + x_pose[:,:3,3].unsqueeze(-1)         # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            # camloc1 = camloc1 / torch.norm(camloc1, dim=-2).reshape(-1, 1, 1)


            # camloc1 = randrot[:, :3, :3]@origin + randrot[:,:3,3].unsqueeze(-1)         # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            # camloc1 = camloc1 / torch.norm(camloc1, dim=-2).reshape(-1, 1, 1)

            '''
            camloc1 = rotmat1@origin    # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            radian1 = self.loc2rad(camloc1) 


            # camloc2 = rotmat2@origin
            # radian2 = self.loc2rad(camloc2) 

            #pdb.set_trace()
            uvs_full = (radian1, radian1.flip(0), randrad)#gt, swap    3, 16, 2
            '''
        '''
        # edit mira start
        if mode == 'val':
            # metric values 
            psnr, ssim = 0, 0
            x_real_np1 = np.array(x_real.detach().cpu())
            image_fake_np1 = np.array(image_fake.detach().cpu())

            # x_real_np2 = np.array(x_real[1].detach().cpu())
            # image_fake_np2 = np.array(image_fake2.detach().cpu())


            for idx in range(len(x_real)):
                x_real_idx1 = np.transpose(x_real_np1[idx], (1, 2, 0))
                image_fake_idx1 = np.transpose(image_fake_np1[idx], (1, 2, 0))
                psnr += compare_psnr(x_real_idx1, image_fake_idx1, data_range=1)
                ssim += compare_ssim(x_real_idx1, image_fake_idx1, multichannel=True, data_range=1)

                # x_real_idx2 = np.transpose(x_real_np2[idx], (1, 2, 0))
                # image_fake_idx2 = np.transpose(image_fake_np2[idx], (1, 2, 0))
                # psnr += compare_psnr(x_real_idx2, image_fake_idx2, data_range=1)
                # ssim += compare_ssim(x_real_idx2, image_fake_idx2, multichannel=True, data_range=1)


            # psnr, ssim = psnr/len(x_real[0])/2, ssim/len(x_real[0])/2
            psnr, ssim = psnr/len(x_real), ssim/len(x_real)


            # img_uint8 = (image_rand * 255).cpu().numpy().astype(np.uint8)
            # img_rand_fid = torch.from_numpy(img_uint8).float() / 255.
            # mu, sigma = calculate_activation_statistics(img_rand_fid)
            
            if val_idx == True:
                out_file_name = f'visualization_{it}_evaluation_P{round(psnr, 2)}_S{round(ssim, 2)}.png'
            else:
                return None, psnr, ssim
        else:
        '''
        out_file_name = 'visualization_%010d.png' % it
        psnr, ssim, fid = 0, 0, 0
        # edit mira end                 
        # image_grid = make_grid(torch.cat((x_real[0].to(self.device), x_real[1].to(self.device), image_fake.clamp_(0., 1.), image_fake2.clamp_(0., 1.), image_swap.clamp_(0., 1.)), dim=0), nrow=image_fake.shape[0])
        image_grid = make_grid(torch.cat((x_real.to(self.device), image_fake.clamp_(0., 1.), shape_rgb.clamp_(0., 1.), appearance_rgb.clamp_(0., 1.), swap_rgb.clamp_(0., 1.)), dim=0), nrow=image_fake.shape[0])

        if mode == 'val':
            save_image(image_grid, os.path.join(self.val_vis_dir, out_file_name))
            # self.record_uvs(uvs_full, os.path.join(self.val_vis_dir), it)
        else:
            save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
            # self.record_uvs(uvs_full, os.path.join(self.vis_dir), it)
        return image_grid, psnr, ssim
