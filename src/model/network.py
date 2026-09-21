import torch.nn as nn

from src.model.fusion import BottleneckFusion
from src.model.tokenizer import ConvTokenizer


class GestureEncoder(nn.Module):
    """EMG + IMU windows -> 256-d representation -> class logits.

    The representation is the two branch means concatenated: the first `d_model` values are the
    muscle branch, the last `d_model` the motion branch. The shared tokens are a conduit and are
    not part of it.
    """

    def __init__(self, num_classes, emg_channels=9, imu_channels=6, d_model=128, num_heads=4,
                 num_layers=4, ffn_dim=256, dropout=0.1, n_bottleneck=32, fusion_layer=2):
        super().__init__()
        self.emg_tok = ConvTokenizer(emg_channels, d_model, dropout)
        self.imu_tok = ConvTokenizer(imu_channels, d_model, dropout)
        self.fuser = BottleneckFusion(d_model, num_heads, num_layers, ffn_dim, dropout,
                                      n_bottleneck, fusion_layer)
        self.head = nn.Linear(2 * d_model, num_classes)

    def extract_features(self, batch):
        e = self.emg_tok(batch["emg"].transpose(1, 2))
        i = self.imu_tok(batch["imu"].transpose(1, 2))
        return self.fuser(e, i)

    def forward(self, batch):
        return self.head(self.extract_features(batch))
