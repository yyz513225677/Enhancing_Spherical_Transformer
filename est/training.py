import os
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .datasets import ClusteredSegmentDataset, KittiSequenceDataset
from .losses import ohem_cross_entropy
from .scheduler import CosineWarmup


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    points = torch.cat([s["points"] for s in samples], dim=0)
    features = torch.cat([s["features"] for s in samples], dim=0)
    labels = torch.cat([s["labels"] for s in samples], dim=0)
    indices = None
    if "indices" in samples[0]:
        indices = torch.cat([s["indices"] for s in samples], dim=0)
    return {"points": points, "features": features, "labels": labels, "indices": indices}


def build_dataloaders(
    data_root: str,
    cluster_root: str,
    sequences: List[str],
    batch_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_ds = KittiSequenceDataset(data_root, sequences)
    cluster_ds = ClusteredSegmentDataset(cluster_root, sequences, focus="clusters")
    other_ds = ClusteredSegmentDataset(cluster_root, sequences, focus="others")

    base_loader = DataLoader(base_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cluster_loader = DataLoader(cluster_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    other_loader = DataLoader(other_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return base_loader, cluster_loader, other_loader


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmup,
    device: torch.device,
    loader_clusters: DataLoader,
    loader_others: DataLoader,
    epoch: int,
    total_epochs: int,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    curriculum_ratio = min(1.0, (epoch + 1) / total_epochs)
    # start with clusters, then mix others
    loaders = [loader_clusters]
    if curriculum_ratio > 0.5:
        loaders.append(loader_others)

    for loader in loaders:
        for batch in loader:
            points = batch["points"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(points, features)
                logits = output["logits"]
                logits = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = ohem_cross_entropy(logits, labels_flat)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def validate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            output = model(points, features)
            logits = output["logits"]
            preds = logits.argmax(dim=-1).view(-1)
            labels_flat = labels.view(-1)
            for cls in range(num_classes):
                pred_mask = preds == cls
                label_mask = labels_flat == cls
                intersection[cls] += (pred_mask & label_mask).sum()
                union[cls] += (pred_mask | label_mask).sum()
    iou = intersection / (union + 1e-6)
    miou = iou.mean().item()
    return {"mIoU": miou, "per_class_iou": iou.detach().cpu().tolist()}


def save_checkpoint(state: Dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
