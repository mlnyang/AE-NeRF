from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
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
                 overwrite_visualization=True, batch_size=None, recon_weight=None, **kwargs):

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
        self.vis_dict = model.generator.get_vis_dict(batch_size)
        self.recon_weight = recon_weight

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
        loss_gen, loss_recon, swap_g = self.train_step_generator(data, it)
        loss_d, reg_d, real_d, swap_d = self.train_step_discriminator(data, it)

        return {
            'gen_total': loss_gen,
            'recon': loss_recon,
            'swap_g': swap_g,
            'disc_total': loss_d,
            'regularizer': reg_d,
            'real_d': real_d,
            'swap_d': swap_d,
        }

    def eval_step(self):
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

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
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

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        if self.multi_gpu:
            latents = generator.module.get_vis_dict(x_real.shape[0])
            # x_fake, x_fake2, x_swap = generator(x_real, pose_real, **latents)        # pred, swap, rand
            x_fake, x_swap = generator(x_real, pose_real, **latents)        # pred, swap, rand

        else:
            # x_fake, x_fake2, x_swap = generator(x_real, pose_real)
            x_fake, x_swap = generator(x_real, pose_real)


        # d_fake = discriminator(x_fake)
        d_swap = discriminator(x_swap)
        # gloss = compute_bce(d_fake, 1)
        gloss_swap = compute_bce(d_swap, 1)

        loss_recon = self.recon_loss(x_fake, x_real) * self.recon_weight
        # loss_new = self.recon_loss(x_fake2, x_real[1].to(self.device)) * self.recon_weight

        gen_loss = loss_recon + gloss_swap
        # gen_loss = (loss_new + loss_recon)/2 
        # gen_loss = loss_recon

        gen_loss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gen_loss.item(), loss_recon.item(), gloss_swap.item()
        # return gen_loss.item(), loss_recon.item()


    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict(batch_size=x_real.shape[0])
                x_swap = generator(x_real, pose_real, **latents)[1]
            else:
                x_swap = generator(x_real, pose_real)[1]

        x_swap.requires_grad_()
        d_swap = discriminator(x_swap)

        d_loss_swap = compute_bce(d_swap, 0)

        loss_d_full += d_loss_swap

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_real + d_loss_swap)

        return (
            d_loss.item(), reg.item(), d_loss_real.item(), d_loss_swap.item())

    # def record_uvs(self, uv, path, it):
    #     out_path = os.path.join(path, 'uv.txt')
    #     name_dict = {0: 'pred', 1:'swap', 2:'rand'}
    #     if not os.path.exists(out_path):
    #         f = open(out_path, 'w')
    #     else:
    #         f = open(out_path, 'a')

    #     for i in range(len(uv)):        # len: 3
    #         line = list(map(lambda x: round(x, 3), uv[i].flatten().detach().cpu().numpy().tolist()))
    #         out = []
    #         for idx in range(0, len(line)//2):
    #             out.append(tuple((line[2*idx], line[2*idx+1])))
    #         txt_line = f'{it}th {name_dict[i]}-uv : {out}\n'
    #         f.write(txt_line)
    #     f.write('\n')
    #     f.close()


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

    # #'ㅅ'
    # def loc2rad(self, loc):
    #     theta = torch.acos(loc[:, 2])
    #     phi = torch.acos(loc[:,0]/torch.sin(theta))
    #     theta, phi = theta * 180 / torch.pi, phi * 180 / torch.pi
    #     return torch.cat([theta, phi], dim=-1)

    # edit
    def loc2rad(self, loc):
        phi = torch.acos(loc[:, 2])
        theta = torch.acos(loc[:,0]/torch.sin(phi))
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
            image_fake, image_swap, image_rand, uvs, radius = self.generator(x_real, x_pose, mode='val', need_uv=True)
            image_fake, image_swap, image_rand = image_fake.detach(), image_swap.detach(), image_rand.detach()

            # edit mira end 


            # import pdb 
            # pdb.set_trace()
            # xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[     # xyz를 self.poses로 rotate -> transpose했던거에 다시 연산됨..! -> 헐 그러면 이 상태에서 예측하는건가보다 그러면 이게 sampled points고 앞에 camera가 원래 ray에서...!
            #                                                                             # 그래야 transformation이 말이 되는 듯.. 근데 그럴거면 왜 transformation을 예측하지..? 
            #     ..., 0                                                                  # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!)
            # ]
            # # 오키.. def encoder에서 생긴 얘가 여기로 들어감!
            # xyz = xyz_rot + self.poses[:, None, :3, 3]      # 얘네가 sampling points!     # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!) 



            # 여기 안을 잘 조정하면 -> swapped view에서도 비슷한 맥락으로 나올 듯 
            # edit 'ㅅ'
            randrad = self.uv2rad(uvs).cuda()        # uv를 radian으로 표현 
            rotmat1 = x_pose[:,:3,:3]        # x_pose : real pose that includes R,t
            # rotmat2 = x_pose[:, 1][:,:3,:3]        # x_pose : real pose that includes R,t
            # import pdb 
            # pdb.set_trace()
            origin = torch.Tensor([0,0,1]).to(self.device).repeat(int(len(x_pose)),1).unsqueeze(-1)

            # translation을 먼저 함 
            # camloc1 = rotmat1@(origin+x_pose[:,:3,3].unsqueeze(-1))   # 얘는 x,y 둘다 못건짐 
            # camloc1 = rotmat1@origin + x_pose[:,:3,3].unsqueeze(-1)         # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            # camloc1 = camloc1 / torch.norm(camloc1, dim=-2).reshape(-1, 1, 1)

            camloc1 = rotmat1@origin    # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            radian1 = self.loc2rad(camloc1) 

            # camloc2 = rotmat2@origin
            # radian2 = self.loc2rad(camloc2) 

            #pdb.set_trace()
            uvs_full = (radian1, radian1.flip(0), randrad)#gt, swap    3, 16, 2


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
            out_file_name = 'visualization_%010d.png' % it
            psnr, ssim, fid = None, None, None
        # edit mira end 
        # image_grid = make_grid(torch.cat((x_real[0].to(self.device), x_real[1].to(self.device), image_fake.clamp_(0., 1.), image_fake2.clamp_(0., 1.), image_swap.clamp_(0., 1.)), dim=0), nrow=image_fake.shape[0])
        image_grid = make_grid(torch.cat((x_real.to(self.device), image_fake.clamp_(0., 1.), image_swap.clamp_(0., 1.), image_rand.clamp_(0., 1.)), dim=0), nrow=image_fake.shape[0])

        if mode == 'val':
            save_image(image_grid, os.path.join(self.val_vis_dir, out_file_name))
            self.record_uvs(uvs_full, os.path.join(self.val_vis_dir), it)
        else:
            save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
            self.record_uvs(uvs_full, os.path.join(self.vis_dir), it)
        return image_grid, psnr, ssim
