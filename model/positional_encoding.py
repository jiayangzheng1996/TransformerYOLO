import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feat=64, temperature=10000, normalize=False, scale=None):
        super(PositionalEncoding, self).__init__()
        self.num_pos_feat = num_pos_feat
        self.temperature = temperature
        self.normalize = normalize
        if normalize:
            if scale is None:
                scale = 2 * math.pi
        self.scale = scale

    def forward(self, inputs: Tensor):
        B, _, H, W = inputs.size()
        y_embedding = torch.ones((B, H, W), device=inputs.device)
        x_embedding = torch.ones((B, H, W), device=inputs.device)

        y_embedding = y_embedding.cumsum(1, dtype=torch.float32)
        x_embedding = x_embedding.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embedding = y_embedding / (y_embedding[:, -1:, :] + eps) * self.scale
            x_embedding = x_embedding / (x_embedding[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feat, dtype=torch.float32, device=inputs.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feat)

        x_pos = x_embedding[:, :, :, None] / dim_t
        y_pos = y_embedding[:, :, :, None] / dim_t
        x_pos = torch.stack((x_pos[:, :, :, 0::2].sin(), x_pos[:, :, :, 1::2].cos()), dim=4).flatten(3)
        y_pos = torch.stack((y_pos[:, :, :, 0::2].sin(), y_pos[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((y_pos, x_pos), dim=3).permute(0, 3, 1, 2)

        return pos


if __name__ == '__main__':
    zeros = torch.zeros((3, 5, 5))
    p = PositionalEncoding()
    encoding = p(zeros)
