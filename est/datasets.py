import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_points(bin_path: str) -> np.ndarray:
    raw = np.fromfile(bin_path, dtype=np.float32)
    channel = 5 if raw.size % 5 == 0 else 4
    points = raw.reshape(-1, channel)
    return points


def load_labels(label_path: str) -> np.ndarray:
    return np.fromfile(label_path, dtype=np.uint32)


def save_split(points: np.ndarray, labels: np.ndarray, indices: np.ndarray, out_bin: str, out_label: str, out_indices: str) -> None:
    points.astype(np.float32).tofile(out_bin)
    labels.astype(np.uint32).tofile(out_label)
    indices.astype(np.int64).tofile(out_indices)


class KittiSequenceDataset(Dataset):
    """Loads KITTI/RELLIS style sequences with aligned labels."""

    def __init__(self, data_root: str, sequences: List[str], transform=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.sequences = sequences
        self.transform = transform
        self.frames: List[Tuple[str, str]] = []
        for seq in sequences:
            velo_dir = os.path.join(data_root, seq, "velodyne")
            label_dir = os.path.join(data_root, seq, "labels")
            if not os.path.isdir(velo_dir):
                continue
            for fname in sorted(os.listdir(velo_dir)):
                if not fname.endswith(".bin"):
                    continue
                stem = fname.replace(".bin", "")
                bin_path = os.path.join(velo_dir, fname)
                label_path = os.path.join(label_dir, f"{stem}.label")
                if os.path.exists(label_path):
                    self.frames.append((bin_path, label_path))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bin_path, label_path = self.frames[idx]
        pts = load_points(bin_path)
        labels = load_labels(label_path)
        assert pts.shape[0] == labels.shape[0], "Point/label size mismatch"
        xyz = torch.from_numpy(pts[:, :3]).float()
        extra = pts[:, 3:]
        features = torch.from_numpy(extra).float()
        labels_t = torch.from_numpy(labels.astype(np.int64))
        sample = {"points": xyz, "features": features, "labels": labels_t, "bin_path": bin_path}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ClusteredSegmentDataset(Dataset):
    """Domain balanced dataset mixing cluster segments and the residual others."""

    def __init__(self, cluster_root: str, sequences: List[str], focus: str = "clusters") -> None:
        super().__init__()
        self.samples: List[Tuple[str, str, str]] = []
        subdir = "clusters" if focus == "clusters" else "others"
        for seq in sequences:
            seq_root = os.path.join(cluster_root, subdir, seq)
            if not os.path.isdir(seq_root):
                continue
            for fname in sorted(os.listdir(seq_root)):
                if not fname.endswith(".bin"):
                    continue
                stem = fname.replace(".bin", "")
                bin_path = os.path.join(seq_root, fname)
                label_path = os.path.join(seq_root, f"{stem}.label")
                index_path = os.path.join(seq_root, f"{stem}.indices")
                if os.path.exists(label_path) and os.path.exists(index_path):
                    self.samples.append((bin_path, label_path, index_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bin_path, label_path, index_path = self.samples[idx]
        pts = load_points(bin_path)
        labels = load_labels(label_path)
        indices = np.fromfile(index_path, dtype=np.int64)
        assert len(pts) == len(labels) == len(indices)
        xyz = torch.from_numpy(pts[:, :3]).float()
        features = torch.from_numpy(pts[:, 3:]).float()
        labels_t = torch.from_numpy(labels.astype(np.int64))
        indices_t = torch.from_numpy(indices.astype(np.int64))
        return {
            "points": xyz,
            "features": features,
            "labels": labels_t,
            "indices": indices_t,
            "bin_path": bin_path,
        }
