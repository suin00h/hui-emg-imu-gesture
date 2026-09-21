import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.model.network import GestureEncoder

N_ELEC = 9


class GestureModule(pl.LightningModule):
    """Classification, optionally with the masked-electrode reconstruction term.

    The reconstruction is optimised jointly with the classification loss; there is no separate
    pretraining stage, and the head is unused at inference.
    """

    def __init__(self, num_classes, emg_channels=9, imu_channels=6, lr=1e-3, weight_decay=1e-2,
                 max_epochs=100, warmup_epochs=5, masked_electrode=False, n_mask=1,
                 aux_weight=1.0, model_kwargs=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = GestureEncoder(num_classes, emg_channels, imu_channels,
                                    **(model_kwargs or {}))
        self.mask_head = nn.Linear(self.model.head.in_features, N_ELEC) if masked_electrode else None
        self.n_classes = num_classes
        self._val = []

    def forward(self, batch):
        return self.model(batch)

    def _reconstruction_loss(self, batch):
        emg = batch["emg"]
        keep = torch.ones(emg.shape[0], 1, N_ELEC, device=emg.device)
        idx = torch.rand(emg.shape[0], N_ELEC, device=emg.device).argsort(1)[:, :self.hparams.n_mask]
        keep.scatter_(2, idx.unsqueeze(1), 0.0)
        target = torch.sqrt((emg ** 2).mean(1) + 1e-8)
        pred = self.mask_head(self.model.extract_features(
            {"emg": emg * keep, "imu": batch["imu"]}))
        m = 1.0 - keep.squeeze(1)
        return ((pred - target) ** 2 * m).sum() / m.sum().clamp_min(1.0)

    def _step(self, batch):
        logits = self.model(batch)
        loss = nn.functional.cross_entropy(logits, batch["label"])
        if self.training and self.mask_head is not None:
            loss = loss + self.hparams.aux_weight * self._reconstruction_loss(batch)
        return logits, loss

    def training_step(self, batch, _):
        loss = self._step(batch)[1]
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        logits, loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self._val.append((logits.argmax(1).cpu(), batch["label"].cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        p = torch.cat([a for a, _ in self._val]).numpy()
        y = torch.cat([b for _, b in self._val]).numpy()
        self.log("val/f1", f1_score(y, p, average="macro",
                                    labels=list(range(self.n_classes)), zero_division=0))
        self._val.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        warm, total = self.hparams.warmup_epochs, self.hparams.max_epochs

        def lr_lambda(epoch):
            if epoch < warm:
                return (epoch + 1) / max(1, warm)
            p = (epoch - warm) / max(1, total - warm)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                                 "interval": "epoch"}}

    @torch.no_grad()
    def fused_features(self, emg, imu, batch=512, device="cpu"):
        self.eval().to(device)
        out = []
        for s in range(0, len(emg), batch):
            out.append(self.model.extract_features({
                "emg": torch.from_numpy(emg[s:s + batch]).to(device),
                "imu": torch.from_numpy(imu[s:s + batch]).to(device)}).cpu().numpy())
        return np.concatenate(out)
