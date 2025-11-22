import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import KittiSequenceDataset, save_split


class DGCNNFeature(nn.Module):
    """Lightweight DGCNN-style edge feature extractor for clustering."""

    def __init__(self, in_channels: int = 4, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

    def get_graph_feature(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        with torch.no_grad():
            inner = -2 * torch.matmul(x, x.transpose(2, 1))
            xx = torch.sum(x ** 2, dim=2, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # [B, N, k]
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.view(B * N, C)
        feature = x[idx, :].view(B, N, self.k, C)
        x = x.view(B, N, 1, C).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)  # batch of 1
        feat = self.get_graph_feature(x)
        feat = feat.permute(0, 2, 3, 1)  # B, N, k, C
        feat = self.mlp(feat)
        return feat.mean(dim=2)  # B, N, C


def cluster_points(points: np.ndarray, features: np.ndarray, distance_threshold: float = 0.4) -> List[np.ndarray]:
    tensor = torch.from_numpy(np.concatenate([points[:, :3], features], axis=1)).float()
    model = DGCNNFeature(in_channels=tensor.shape[1])
    with torch.no_grad():
        edge_feat = model(tensor)  # 1, N, C
    feat = edge_feat.squeeze(0)
    coords = torch.from_numpy(points[:, :3]).float()
    neighbors = []
    for i in range(points.shape[0]):
        dist = torch.norm(coords - coords[i], dim=1)
        mask = dist < distance_threshold
        neighbors.append(mask)
    visited = torch.zeros(points.shape[0], dtype=torch.bool)
    clusters = []
    for idx in range(points.shape[0]):
        if visited[idx]:
            continue
        queue = [idx]
        visited[idx] = True
        current = [idx]
        while queue:
            cur = queue.pop()
            neigh = neighbors[cur]
            candidate = torch.where(neigh & ~visited)[0]
            for c in candidate.tolist():
                visited[c] = True
                queue.append(c)
                current.append(c)
        clusters.append(np.array(current, dtype=np.int64))
    return clusters


def run_offline_clustering(data_root: str, output_root: str, sequences: List[str]) -> None:
    os.makedirs(output_root, exist_ok=True)
    dataset = KittiSequenceDataset(data_root, sequences)
    for bin_path, label_path in dataset.frames:
        pts = np.fromfile(bin_path, dtype=np.float32)
        channel = 5 if pts.size % 5 == 0 else 4
        points = pts.reshape(-1, channel)
        labels = np.fromfile(label_path, dtype=np.uint32)
        features = points[:, 3:]
        clusters = cluster_points(points, features)
        stem = os.path.basename(bin_path).replace(".bin", "")
        seq = bin_path.split(os.sep)[-3]
        cluster_dir = os.path.join(output_root, "clusters", seq)
        other_dir = os.path.join(output_root, "others", seq)
        os.makedirs(cluster_dir, exist_ok=True)
        os.makedirs(other_dir, exist_ok=True)

        kept_mask = np.zeros(points.shape[0], dtype=bool)
        for cid, idxs in enumerate(clusters):
            kept_mask[idxs] = True
            cluster_points_arr = points[idxs]
            cluster_labels_arr = labels[idxs]
            out_bin = os.path.join(cluster_dir, f"{stem}_{cid:03d}.bin")
            out_label = os.path.join(cluster_dir, f"{stem}_{cid:03d}.label")
            out_indices = os.path.join(cluster_dir, f"{stem}_{cid:03d}.indices")
            save_split(cluster_points_arr, cluster_labels_arr, idxs, out_bin, out_label, out_indices)

        other_idxs = np.where(~kept_mask)[0]
        other_points_arr = points[other_idxs]
        other_labels_arr = labels[other_idxs]
        out_bin = os.path.join(other_dir, f"{stem}_others.bin")
        out_label = os.path.join(other_dir, f"{stem}_others.label")
        out_indices = os.path.join(other_dir, f"{stem}_others.indices")
        save_split(other_points_arr, other_labels_arr, other_idxs, out_bin, out_label, out_indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline DGCNN clustering")
    parser.add_argument("data_root", type=str, help="Root directory with KITTI/RELLIS sequences")
    parser.add_argument("output_root", type=str, help="Where to store clustered splits")
    parser.add_argument("--sequences", nargs="*", default=[f"{i:02d}" for i in range(11)])
    args = parser.parse_args()
    run_offline_clustering(args.data_root, args.output_root, args.sequences)
