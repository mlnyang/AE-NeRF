from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.stylegan2_layers import ConvLayer, ResBlock, EqualLinear
import torch.nn.functional as F


class StyleGAN2PatchDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(StyleGAN2PatchDiscriminator, self).__init__()
        self.netPatchD_scale_capacity = kwargs['netPatchD_scale_capacity']
        self.netPatchD_max_nc = kwargs['netPatchD_max_nc']
        self.patch_size = kwargs['patch_size']
        self.max_num_tiles = kwargs['max_num_tiles']
        self.use_antialias = kwargs['use_antialias']

        self.spatial_code_ch = kwargs['spatial_code_ch']
        self.global_code_ch = kwargs['global_code_ch']
        self.lambda_R1 = kwargs['lambda_R1']
        self.lambda_patch_R1 = kwargs['lambda_patch_R1']
        self.lambda_L1 = kwargs['lambda_L1']
        self.lambda_GAN = kwargs['lambda_GAN']
        self.lambda_PatchGAN = kwargs['lambda_PatchGAN']
        self.patch_min_scale = kwargs['patch_min_scale']
        self.patch_max_scale = kwargs['patch_max_scale']
        self.patch_num_crops = kwargs['patch_num_crops']
        self.patch_use_aggregation = kwargs['patch_use_aggregation']


        channel_multiplier = self.netPatchD_scale_capacity      # default: 4.0
        size = self.patch_size
        channels = {
            4: min(self.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))
        in_channel = channels[2 ** log_size]
        blur_kernel = [1, 3, 3, 1] if self.use_antialias else [1]

        convs = [('0', ConvLayer(3, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ResBlock(in_channel, self.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.netPatchD_max_nc * 2, self.netPatchD_max_nc, 3, pad=0)))
                                                                        # netPatchD_max_nc : 256 + 128
        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out

        

    def forward(self, real, fake, fake_only=False):
        assert real is not None
        real_patches, patch_ids = self.sample_patches(real, None)
        if fake is None:
            real_patches.requires_grad_()
        real_feat = self.extract_features(real_patches)

        bs = real.size(0)
        if fake is None or not fake_only:
            pred_real = self.discriminate_features(
                real_feat,
                torch.roll(real_feat, 1, 1))
            pred_real = pred_real.view(bs, -1)


        if fake is not None:
            fake_patches = self.sample_patches(fake, patch_ids)
            fake_feat = self.extract_features(fake_patches)
            pred_fake = self.discriminate_features(
                real_feat,
                torch.roll(fake_feat, 1, 1))
            pred_fake = pred_fake.view(bs, -1)

        if fake is None:
            return pred_real, real_patches
        elif fake_only:
            return pred_fake
        else:
            return pred_real, pred_fake
