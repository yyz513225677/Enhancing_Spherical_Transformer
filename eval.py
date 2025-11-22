import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from est import ESTransformer
from est.datasets import load_points


def load_segment(bin_path: str, index_path: str):
    points = load_points(bin_path)
    indices = np.fromfile(index_path, dtype=np.int64)
    return points, indices


def merge_predictions(pred_segments: List[np.ndarray], index_segments: List[np.ndarray], total_points: int) -> np.ndarray:
    merged = np.zeros(total_points, dtype=np.int64)
    for preds, idxs in zip(pred_segments, index_segments):
        merged[idxs] = preds
    return merged


def evaluate_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_mask = preds == cls
        label_mask = labels == cls
        intersection[cls] = np.logical_and(pred_mask, label_mask).sum()
        union[cls] = np.logical_or(pred_mask, label_mask).sum()
    iou = intersection / (union + 1e-6)
    return {"mIoU": float(iou.mean()), "per_class_iou": iou.tolist()}


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESTransformer(num_classes=args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for seq in args.sequences:
            cluster_dir = os.path.join(args.cluster_root, "clusters", seq)
            other_dir = os.path.join(args.cluster_root, "others", seq)
            if not os.path.isdir(cluster_dir):
                continue
            cluster_bins = sorted([f for f in os.listdir(cluster_dir) if f.endswith(".bin")])
            frames = {}
            for fname in cluster_bins:
                stem = fname.replace(".bin", "")
                frame_id = stem.split("_")[0]
                frames.setdefault(frame_id, {"clusters": [], "others": None})
                if "others" in stem:
                    continue
                bin_path = os.path.join(cluster_dir, fname)
                label_path = os.path.join(cluster_dir, f"{stem}.label")
                index_path = os.path.join(cluster_dir, f"{stem}.indices")
                frames[frame_id]["clusters"].append((bin_path, label_path, index_path))
            if os.path.isdir(other_dir):
                for fname in os.listdir(other_dir):
                    if not fname.endswith(".bin"):
                        continue
                    stem = fname.replace(".bin", "")
                    frame_id = stem.split("_")[0]
                    bin_path = os.path.join(other_dir, fname)
                    label_path = os.path.join(other_dir, f"{stem}.label")
                    index_path = os.path.join(other_dir, f"{stem}.indices")
                    if frame_id in frames:
                        frames[frame_id]["others"] = (bin_path, label_path, index_path)

            for frame_id, content in frames.items():
                segment_preds = []
                segment_indices = []
                segment_labels = []
                total_points = 0
                for bin_path, label_path, index_path in content["clusters"]:
                    points, indices = load_segment(bin_path, index_path)
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    total_points = max(total_points, indices.max() + 1)
                    pts_tensor = torch.from_numpy(points[:, :3]).float().to(device)
                    feats_tensor = torch.from_numpy(points[:, 3:]).float().to(device)
                    out = model(pts_tensor, feats_tensor)
                    pred = out["logits"].argmax(dim=-1).cpu().numpy()
                    segment_preds.append(pred)
                    segment_indices.append(indices)
                    segment_labels.append(labels)

                if content["others"] is not None:
                    bin_path, label_path, index_path = content["others"]
                    points, indices = load_segment(bin_path, index_path)
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    total_points = max(total_points, indices.max() + 1)
                    pts_tensor = torch.from_numpy(points[:, :3]).float().to(device)
                    feats_tensor = torch.from_numpy(points[:, 3:]).float().to(device)
                    out = model(pts_tensor, feats_tensor)
                    pred = out["logits"].argmax(dim=-1).cpu().numpy()
                    segment_preds.append(pred)
                    segment_indices.append(indices)
                    segment_labels.append(labels)

                merged_preds = merge_predictions(segment_preds, segment_indices, total_points)
                merged_labels = np.zeros_like(merged_preds)
                for lbl, idxs in zip(segment_labels, segment_indices):
                    merged_labels[idxs] = lbl
                metrics = evaluate_metrics(merged_preds, merged_labels, args.num_classes)
                all_metrics.append(metrics)
                print(f"Seq {seq} frame {frame_id}: mIoU={metrics['mIoU']:.4f}")

    if all_metrics:
        mean_miou = sum(m["mIoU"] for m in all_metrics) / len(all_metrics)
        per_class = np.mean([m["per_class_iou"] for m in all_metrics], axis=0).tolist()
        print({"mIoU": mean_miou, "per_class_iou": per_class})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EST on clustered splits")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("cluster_root", type=str)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--sequences", nargs="*", default=[f"{i:02d}" for i in range(11)])
    args = parser.parse_args()
    run_inference(args)
