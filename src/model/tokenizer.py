import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """[B, C, L] -> [B, N, d]. Stride-2 convolutions; 150 samples become 16 tokens.

    BatchNorm here is the only place signal statistics are normalised inside the network, and its
    running statistics are estimated on the source subjects. Run the model in eval() when extracting
    features or the held-out person's own batch statistics leak in.
    """

    def __init__(self, in_channels, d_model=128, dropout=0.1):
        super().__init__()
        layers, c = [], in_channels
        for i, oc in enumerate([32, 64, d_model]):
            layers += [nn.Conv1d(c, oc, kernel_size=5, stride=2), nn.BatchNorm1d(oc)]
            if i < 2:
                layers.append(nn.GELU())
            c = oc
        self.conv = nn.Sequential(*layers)
        self.type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.conv(x).transpose(1, 2) + self.type_embed)
