import torch
from torch import nn, Tensor
from torch.nn import Module

from positional_encoding import PositionalEncoding


class PointwiseFeedForwardNetwork(Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class ViT(Module):
    def __init__(self, image_res, patch_res, latent_dim, nchannels: int = 3):
        super(ViT, self).__init__()
        assert image_res % patch_res == 0, "Image resolution must be divisible by patch resolution."
        self.image_res = image_res
        self.patch_res = patch_res

        self.npatches = (image_res // patch_res)**2
        self.patch_dim = nchannels * patch_res**2
        self.patch_embedding = nn.Linear(self.patch_dim, latent_dim)

        self.pe = PositionalEncoding(d_model=latent_dim, max_len=nchannels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, x: Tensor):
        r"""

        Args:
            x: (bsz, C, H, W)

        Returns:

        """
        x = x.transpose(1, -1).view(-1, self.npatches, self.patch_dim)
        x = self.patch_embedding(x)


