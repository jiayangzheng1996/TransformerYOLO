import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

from .transformer import Transformer
from .backbone import Backbone
from .positional_encoding import PositionalEncoding


class Detector(nn.Module):
    def __init__(self, num_classes, num_grid, num_boxes, args, hidden_dim=256, n_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Detector, self).__init__()
        self.num_grid = num_grid
        self.backbone = Backbone('resnet50', args.train_backbone, False, args.dilation)

        # self.input_proj = nn.Conv2d(resnet50().fc.in_features, hidden_dim, 1)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)
        self.query = nn.Embedding(num_grid * num_grid, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_heads, num_encoder_layers, num_decoder_layers)

        self.fc_layer = nn.Linear(hidden_dim, 5 * num_boxes + num_classes)
        # self.row_encoding = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # self.column_encoding = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.pos_encode = PositionalEncoding(num_pos_feat=hidden_dim//2, normalize=True)

    def forward(self, input):
        x = self.backbone(input)
        h = self.input_proj(x[-1])
        batch = h.shape[0]
        # pos = torch.cat([
        #     self.column_encoding[:W].unsqueeze(0).repeat(H, 1, 1),
        #     self.row_encoding[:H].unsqueeze(1).repeat(1, W, 1)
        # ], dim=-1).flatten(0, 1).unsqueeze(1)
        pos = self.pos_encode(h)
        pos = pos.flatten(2).permute(2, 0, 1)
        h = h.flatten(2).permute(2, 0, 1)
        h = self.transformer(h, pos, self.query.weight.unsqueeze(1).expand(-1, batch, -1)).transpose(0, 1)

        pred = self.fc_layer(h).view(batch, self.num_grid, self.num_grid, -1)
        out = pred.sigmoid()
        return out


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FFN, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * num_layers
        self.layers = nn.ModuleList(nn.Linear(in_dim, out_dim)
                                    for in_dim, out_dim in zip([input_dim] + h, h + [output_dim]))

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(inputs)) if i < self.num_layers - 1 else layer(x)
        return x

if __name__ == '__main__':
    model = Detector(10, 100)
    input = torch.rand(32, 3, 227, 227)
    output = model(input)