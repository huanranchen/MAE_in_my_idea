import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size = 96, patch_size = 16, dim=768,mlp_dim = 256, depth=6,
                 num_heads = 2, dropout = 0.2, num_classes=96*96*3
                 ):
        super(ViT, self).__init__()
        assert not image_size%patch_size, "in ViT, image_size cannot divided by patch_size"

        self.patch_size = patch_size
        self.patches_each_dim = image_size // patch_size
        self.dim = dim
        self.transformer = nn.TransformerEncoderLayer(
            d_model=dim, dim_feedforward=mlp_dim, dropout=dropout, nhead=2
        )
        self.to_token = nn.Linear(int(3*patch_size*patch_size), dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.num_patches = self.patches_each_dim**2

    def forward(self, x):
        #我这里只支持H=W的
        N, C, H, W = x.shape
        # batch N,  然后K个patch
        patches = x.view(N, C, self.patches_each_dim, self.patch_size,self.patches_each_dim,
                         self.patch_size).permute(0,2,4,3,5,1).reshape(N, self.patches_each_dim**2,-1)

        # to tokens


        tokens = self.to_token(patches)
        tokens = torch.cat([self.cls_token.repeat(N, 1, 1), tokens], dim=1)


        encodes = self.transformer(tokens)

        #然后和bert一样，只用cls
        x = encodes[:,0,:]
        x = self.mlp(x)

        return x.reshape(N, C, H, W)
