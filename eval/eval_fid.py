
# same as eval_fid.py in cvlab4
print("running")

import sys
import os
import inspect

import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import random
from PIL import Image
import glob
import time
import random
from torchvision.utils import save_image, make_grid
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from im2scene import config
from im2scene.checkpoints import CheckpointIO
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from im2scene import config
from im2scene.checkpoints import CheckpointIO
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
logger_py = logging.getLogger(__name__)
np.random.seed(500)
torch.manual_seed(500)


# 세팅 나중에 잡기!
# random_seed = 0

# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True       # 연산속도 느려짐!
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)



# Arguments
parser = argparse.ArgumentParser(
    description='Train a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--mode', type=str, default="recon", help = "mode"  )


args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# get data
dataset = config.get_dataset(cfg)
len_dset = len(dataset)
train_len = len_dset * 0.9

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(train_len), int(len_dset-train_len)])

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, num_workers=4, shuffle=False,
    pin_memory=True, drop_last=True,
)



model = config.get_model(cfg, device=device, len_dataset=len(train_dataset))
out_dir = cfg['training']['out_dir']
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generate
model.eval()
gen = model.generator
# gen = model.generator_test

img_fake = []


for idx, batch in enumerate(val_loader):
    images = batch.get('image').to(device)
    with torch.no_grad():
        rgb, shape_rgb, appearance_rgb, swap_rgb = gen(images, mode="val")
        rgb, shape_rgb, appearance_rgb, swap_rgb = rgb.detach(), shape_rgb.detach(), appearance_rgb.detach(), swap_rgb.detach()
        for i in range(len(images)):
            if args.mode == "recon":
                img_fake.append(rgb[i])
            elif args.mode == "shape":
                img_fake.append(shape_rgb[i])
            elif args.mode == "appearance":
                img_fake.append(appearance_rgb[i])
            elif args.mode == "pose":
                img_fake.append(swap_rgb[i])
            else:
                raise ValueError("Not appropriate mode")    
                
            

fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


out_dict_file = os.path.join(out_dir, 'fid_evaluation_' + args.mode + '_.npz')
out_img_file = os.path.join(out_dir, 'fid_images_' + args.mode + '_.npy')
out_vis_file = os.path.join(out_dir, 'fid_images_' + args.mode + '_.jpg')
out_txt_file = os.path.join(out_dir,'fid_values_' + args.mode + '_.txt') 


img_fake = torch.stack(img_fake, dim=0).cpu()


img_fake.clamp_(0., 1.)
n_images = img_fake.shape[0]

out_dict = {'n_images': n_images}


img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
np.save(out_img_file, img_uint8)

# use unit for eval to fairy compare
img_fake = torch.from_numpy(img_uint8).float() / 255.
#summary = torch.cat((summary, img_fake[:batch_size]), dim=0)

print("calculating activation statistics")

mu, sigma = calculate_activation_statistics(img_fake)
out_dict["m"] = mu
out_dict["sigma"] = sigma

# calculate FID score and save it to a dictionary
print("calculating frechet distance")
fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
out_dict['fid'] = fid_score
print(args.mode,"FID Score (%d images): %.6f" % (n_images, fid_score))
np.savez(out_dict_file, **out_dict)
with open('readme.txt', 'w') as f:
    f.write(args.mode + " FID Score (%d images): %.6f" % (n_images, fid_score))

# Save a grid of 16x16 images for visualization
print(out_vis_file)
save_image(make_grid(img_fake[:128], nrow=8, pad_value=1.), out_vis_file)


# Make Summary Grid
#summary_vis_file = os.path.join(output_dir, 'fid_images_' + "summary" + '_.jpg')
#save_image(make_grid(summary, nrow=batch_size, pad_value=1.), summary_vis_file)