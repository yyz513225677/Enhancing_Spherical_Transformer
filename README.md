# Enhancing Spherical Transformer (EST)

This repository implements the end-to-end EST pipeline:

```
DGCNN clustering (offline) → exact split of .bin/.label with preserved indices
→ spherical window tokenizer → 3-stage deeper spherical transformer
(pre-norm MHSA + MLP + DropPath) → cross-scale fusion head → classifier
→ OHEM CE loss → clustered curriculum/domain-balanced training
→ per-cluster inference + merge by indices
```

The code targets KITTI/RELLIS-style LiDAR scans stored as
`.bin` float32 `[x, y, z, intensity]` (optionally `ring`) plus `.label` `uint32`
files with identical ordering. All reading/writing uses `numpy.fromfile/tofile`.

## Environment

* Python 3.10+, PyTorch
* CUDA is optional (AMP automatically falls back to CPU).

## Data layout

```
<data_root>/<seq>/velodyne/<frame>.bin
<data_root>/<seq>/labels/<frame>.label
```

Sequences `00–10` are handled by default. Offline clustering writes exact splits
with preserved point indices:

```
<cluster_root>/clusters/<seq>/<frame>_<cluster>.bin|label|indices
<cluster_root>/others/<seq>/<frame>_others.bin|label|indices
```

`indices` files store the original point positions, enabling loss computation
and inference to perfectly align predictions and labels.

## Offline clustering

Run DGCNN-based clustering once to generate curriculum-friendly segments.

```bash
python -m est.cluster /path/to/data /path/to/output --sequences 00 01 02
```

Outputs are created under `clusters/` and `others/` with masks preserved by the
`*.indices` files.

## Training

The trainer enforces batch size 1, cosine LR with warmup, AMP, and gradient
clipping. Curriculum/domain balancing starts with clustered segments then mixes
residual points.

```bash
python train.py /path/to/data /path/to/clustered \
  --epochs 30 --lr 1e-3 --num_classes 20 --warmup_steps 200
```

Checkpoints are saved to `checkpoints/est.pth` by default.

## Evaluation

Inference is performed per cluster and per residual split, then merged by stored
indices for metric computation (mIoU and per-class IoU).

```bash
python eval.py checkpoints/est.pth /path/to/clustered --num_classes 20
```

Per-frame metrics are printed along with averaged summary statistics.
