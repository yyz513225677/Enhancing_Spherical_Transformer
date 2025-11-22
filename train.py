import argparse
import os
from typing import List

import torch
from torch import optim

from est import ESTransformer
from est.training import build_dataloaders, save_checkpoint, train_epoch, validate
from est.scheduler import CosineWarmup


def parse_args():
    parser = argparse.ArgumentParser(description="Train EST spherical transformer")
    parser.add_argument("data_root", type=str, help="Root directory with KITTI/RELLIS sequences")
    parser.add_argument("cluster_root", type=str, help="Directory with clustered splits")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--sequences", nargs="*", default=[f"{i:02d}" for i in range(11)])
    parser.add_argument("--output", type=str, default="checkpoints/est.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_loader, cluster_loader, other_loader = build_dataloaders(
        args.data_root, args.cluster_root, args.sequences, batch_size=args.batch_size
    )

    model = ESTransformer(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(len(cluster_loader), 1)
    scheduler = CosineWarmup(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            optimizer,
            scheduler,
            device,
            cluster_loader,
            other_loader,
            epoch,
            args.epochs,
            scaler,
        )
        metrics = validate(model, device, base_loader, args.num_classes)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, mIoU={metrics['mIoU']:.4f}")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.current_step,
                "scaler_state": scaler.state_dict(),
                "metrics": metrics,
            },
            args.output,
        )


if __name__ == "__main__":
    main()
