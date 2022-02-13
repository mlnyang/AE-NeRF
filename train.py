import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
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

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
lr = cfg['training']['learning_rate']
lr_d = cfg['training']['learning_rate_d']
batch_size = cfg['training']['batch_size']
n_workers = cfg['training']['n_workers']
t0 = time.time()

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

dataset = config.get_dataset(cfg)
len_dset = len(dataset)
train_len = len_dset * 0.9

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(train_len), int(len_dset-train_len)])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
    pin_memory=True, drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
    pin_memory=True, drop_last=True,
)

model = config.get_model(cfg, device=device, len_dataset=len(train_dataset))


# Initialize training
op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
optimizer_kwargs = cfg['training']['optimizer_kwargs']

if hasattr(model, "generator") and model.generator is not None:
    parameters_g = model.generator.parameters()
else:
    parameters_g = list(model.decoder.parameters())
optimizer = op(parameters_g, lr=lr, **optimizer_kwargs)

if hasattr(model, "discriminator") and model.discriminator is not None:
    parameters_d = model.discriminator.parameters()
    optimizer_d = op(parameters_d, lr=lr_d)
else:
    optimizer_d = None

trainer = config.get_trainer(model, optimizer, optimizer_d, cfg, device=device)
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer,
                             optimizer_d=optimizer_d)
try:
    load_dict = checkpoint_io.load('model.pt')
    print("Loaded model checkpoint.")
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
logger_py.info(f'NAME: {cfg["training"]["out_dir"]} \n Major settings: \n 1) LR_G: {cfg["training"]["learning_rate"]} \n 2) LR_D: {cfg["training"]["learning_rate_d"]} \n 3) range_u: {cfg["training"]["range_u"]} \n 4) range_v: {cfg["training"]["range_v"]} \n 5) recon_weight: {cfg["training"]["recon_weight"]}')
# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)

if hasattr(model, "discriminator") and model.discriminator is not None:
    nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
    logger_py.info(
        'Total number of discriminator parameters: %d' % nparameters_d)
if hasattr(model, "generator") and model.generator is not None:
    nparameters_g = sum(p.numel() for p in model.generator.parameters())
    logger_py.info('Total number of generator parameters: %d' % nparameters_g)

t0b = time.time()

while (True):
    epoch_it += 1

    for batch in train_loader:

        it += 1
        loss = trainer.train_step(batch, it)
        for (k, v) in loss.items():
            logger.add_scalar(k, v, it)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (
                epoch_it, it, time.time() - t0b)
            for (k, v) in loss.items():
                info_txt += ', %s: %.4f' % (k, v)
            logger_py.info(info_txt)
            t0b = time.time()

        # # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            logger_py.info('Visualizing')
            image_grid, psnr, ssim = trainer.visualize(batch, it=it, mode='train', val_idx=None)
            if image_grid is not None:
                logger.add_image('images', image_grid, it)

            '''psnr_batch = 0.0
            ssim_batch = 0.0
            # fid_batch = 0.0
            for val_idx, val_batch in enumerate(val_loader):
                image_grid, psnr, ssim = trainer.visualize(val_batch, it=it, mode='val', val_idx=val_idx==len(val_loader)-1)
                if val_idx == len(val_loader)-1:
                    logger.add_image('images', image_grid, it)
                psnr_batch += psnr
                ssim_batch += ssim 
                # fid_batch += fid
            total_psnr = psnr_batch / len(val_loader)
            total_ssim = ssim_batch / len(val_loader)
            # total_fid = fid_batch / len(val_loader)
            print(f'Validation Loss : PSNR {total_psnr} | SSIM {total_ssim} ')'''

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            metric_val_best = np.array([1/(it+1)])
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        '''
        # Run validation            <- validadtion 확인해보기!    # 일단 pass
        if validate_every > 0 and (it % validate_every) == 0 and (it > 0):      # 여기서 validation까지도 전부 계산하기
            print("Performing evaluation step.")
            eval_dict = trainer.evaluate()
            metric_val = eval_dict[model_selection_metric]
            logger_py.info('Validation metric (%s): %.4f'
                           % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            # if model_selection_sign * (metric_val - metric_val_best) > 0:
            #     metric_val_best = metric_val
            #     logger_py.info('New best model (loss %.4f)' % metric_val_best)
            #     checkpoint_io.backup_model_best('model_best.pt')
            #     checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
            #                        loss_val_best=metric_val_best)
        '''



        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            logger_py.info('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
