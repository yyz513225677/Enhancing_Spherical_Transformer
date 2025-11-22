from typing import Dict, List, Tuple

import torch
from torch import nn

from .blocks import TransformerBlock
from .tokenizer import SphericalWindowTokenizer


class CrossScaleFusion(nn.Module):
    def __init__(self, dims: List[int], out_dim: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, out_dim) for d in dims])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        upsampled = []
        for idx, feat in enumerate(features):
            proj_feat = self.proj[idx](feat)
            pooled = proj_feat.mean(dim=1, keepdim=True)
            upsampled.append(pooled)
        fused = torch.cat(upsampled, dim=1).mean(dim=1)
        return self.norm(fused)


class ESTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 64,
        depth: Tuple[int, int, int] = (2, 2, 2),
        heads: Tuple[int, int, int] = (4, 4, 8),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = SphericalWindowTokenizer(feature_dim=feature_dim, embed_dim=embed_dim)

        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            dim=dims[idx],
                            heads=heads[idx],
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path,
                        )
                        for _ in range(depth[idx])
                    ]
                )
                for idx in range(3)
            ]
        )

        self.downsample_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.fusion = CrossScaleFusion(dims, dims[-1])
        self.classifier = nn.Sequential(
            nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes)
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, window_ids = self.tokenizer(points, features)
        tokens = tokens.unsqueeze(0)  # batch size 1 enforced

        stage_outputs: List[torch.Tensor] = []
        token = tokens
        for stage_idx, stage in enumerate(self.stages):
            for block in stage:
                token = block(token)
            stage_outputs.append(token)
            if stage_idx < len(self.downsample_layers):
                token = self.downsample_layers[stage_idx](token)

        fused = self.fusion(stage_outputs)
        logits = self.classifier(fused)
        return {"logits": logits, "window_ids": window_ids}
