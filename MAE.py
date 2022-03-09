import torch
import torch.nn as nn


class MAE(nn.Module):
    def __init__(self, encoder, decoder, image_size=96, patch_size=16, masked_ratio=0.75, ):
        super(MAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = encoder.patch_size
        self.patches_each_dim = encoder.patches_each_dim
        self.masked_ratio = masked_ratio
        self.dim = encoder.dim
        self.patch_size = patch_size
        self.masked_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.image_size = image_size

    def mask(self, x):
        x = x.permute(0, 3, 1, 2)

        N, C, H, W = x.shape
        num_patches = int(self.patches_each_dim ** 2)
        patches = x.view(
            N, C,
            H // self.patch_size, self.patch_size,
            W // self.patch_size, self.patch_size
        ).permute(0, 2, 4, 3, 5, 1)

        shape = patches.shape
        patches = patches.reshape(N, num_patches, -1)
        y = patches.clone()

        num_masked = int(self.masked_ratio * num_patches)
        shuffle_indices = torch.rand(N, num_patches).argsort()

        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
        batch_ind = torch.arange(N).unsqueeze(-1)

        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

        mask_patches[:, :, :] = self.masked_token

        #unshuffle

        concates = torch.cat([mask_patches, unmask_patches],dim=1)
        patches = torch.zeros_like(concates)
        patches[batch_ind,shuffle_indices]=concates



        patches = patches.reshape(shape)
        patches = patches.permute(0, 5, 1, 3, 2, 4).reshape(x.shape)

        return patches

    def encoder_foward(self, x):
        x = self.mask(x)
        return self.encoder.forward(x)

    def decoder_forward(self, x):
        return self.decoder.forward(x)

    def forward(self, x):
        x = self.encoder_foward(x)
        return self.decoder_forward(x).permute(0, 2, 3, 1)
