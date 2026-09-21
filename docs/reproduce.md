# Reproducing the paper

```bash
pip install -r requirements.txt
python scripts/download.py          # dataset/ and checkpoints/
python scripts/verify.py            # checksums and shapes
```

## Table 1 — the benchmark

Using the released features, no GPU needed:

```bash
python scripts/run_benchmark.py --features checkpoints/features --out runs/benchmark.csv
```

From the released checkpoints instead:

```bash
for S in $(seq 1 11); do
  python scripts/extract_features.py --checkpoint checkpoints/ours/S$(printf %02d $S).ckpt \
    --target $S --out runs/ours
done
python scripts/run_benchmark.py --features runs/ours --out runs/benchmark.csv
```

From scratch (one encoder per fold, GPU):

```bash
for S in $(seq 1 11); do
  python scripts/train_encoder.py --config configs/encoder_ours.yaml --target $S
done
```

`train_encoder.py` names its output directory by a hash of the whole config, so two configurations
cannot share a feature cache.

## Table 4 — the ablation

Four configurations, each 11 folds:

| row | config |
|---|---|
| Baseline | `configs/encoder_baseline.yaml` |
| Masked reconstruction | baseline + `masked_electrode` |
| Channel rotation | baseline + `channel_rotation` |
| Both | `configs/encoder_ours.yaml` |

## Figures

```bash
python scripts/make_figures.py
```
