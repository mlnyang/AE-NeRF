from im2scene.discriminator import conv, patch_discriminator, patchgan


discriminator_patchD_dict = {
    'patchD': patchgan.NLayerDiscriminator
}
discriminator_plainD_dict = {
    'dc': conv.DCDiscriminator,
    #'resnet': conv.DiscriminatorResnet
}


patch_discriminator_dict = {
    'patch': patch_discriminator.StyleGAN2PatchDiscriminator
}