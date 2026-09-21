import torch
import torch.nn as nn


class BottleneckFusion(nn.Module):
    """MBT-style fusion (Nagrani et al., NeurIPS 2021).

    Two per-modality transformer stacks. A block of learned shared tokens is concatenated to each
    stream from `fusion_layer` on; each stream reads and writes it in one self-attention, and the
    two writes are averaged. That average is the only cross-modal path.

    The shared tokens do not enter at the input: layers below `fusion_layer` see their own stream
    only. Information needs two exchanging layers to cross -- one to write, a later one to read --
    so `fusion_layer` must be at most `num_layers - 2` or the branches stay independent.
    """

    def __init__(self, d_model, num_heads, num_layers, ffn_dim, dropout,
                 n_bottleneck=32, fusion_layer=2):
        super().__init__()
        if not 0 <= fusion_layer <= num_layers - 2:
            raise ValueError(f"fusion_layer {fusion_layer} must be <= num_layers - 2")
        mk = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.emg_layers = nn.ModuleList([mk() for _ in range(num_layers)])
        self.imu_layers = nn.ModuleList([mk() for _ in range(num_layers)])
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottleneck, d_model) * 0.02)
        self.n_bottleneck = n_bottleneck
        self.fusion_layer = fusion_layer

    def forward(self, emg_tokens, imu_tokens):
        e, i, n = emg_tokens, imu_tokens, self.n_bottleneck
        b = self.bottleneck.expand(e.shape[0], -1, -1)
        for depth, (le, li) in enumerate(zip(self.emg_layers, self.imu_layers)):
            if depth < self.fusion_layer:
                e, i = le(e), li(i)
                continue
            eo, io = le(torch.cat([e, b], 1)), li(torch.cat([i, b], 1))
            e, i = eo[:, :-n], io[:, :-n]
            b = 0.5 * (eo[:, -n:] + io[:, -n:])
        return torch.cat([e.mean(1), i.mean(1)], dim=1)
