import torch.nn as nn

class scorer(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(scorer, self).__init__()
        conv4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding_mode='zeros')

        pool = nn.MaxPool2d(kernel_size=4, ceil_mode=True)
        self.part2 = nn.Sequential(conv4, pool)

    def forward(self, x_low):
        out = self.part2(x_low)
        return out


import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Multi-head Self-Attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        # Feed Forward
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x


class Scorer2(nn.Module):

    def __init__(self, img_size=16, patch_size=4, num_layers=2, dim=512, heads=8, mlp_dim=1024):
        super(Scorer, self).__init__()

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Linear(patch_size * patch_size * 1, dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(dim, 1)

    def forward(self, x_low):
        batch_size, _, height, width = x_low.shape
        x_low = rearrange(x_low, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x_low = self.patch_embedding(x_low)

        for transformer_block in self.transformer_blocks:
            x_low = transformer_block(x_low)

        scores = self.fc(x_low)  # [B, num_patches, 1]

        scores = rearrange(scores, 'b (h w) 1 -> b 1 h w', h=height // self.patch_size,
                                  w=width // self.patch_size)

        return scores
