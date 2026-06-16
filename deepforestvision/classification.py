from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def build_val_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


class DinoV3Head(nn.Module):
    """
    Backbone = DINOv3 ViT.
    Head input = concat([CLS], mean(patch_tokens)) -> dim = 2 * embed_dim
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim  # DINOv3 ViT exposes embed_dim

        self.classifier = (
            nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # get_intermediate_layers returns a list; take last requested layer
        seq = self.backbone.get_intermediate_layers(pixel_values, n=1)[0]  # [B, 1+N, D]

        cls_token = seq[:, 0]          # [B, D]
        patch_tokens = seq[:, 1:]      # [B, N, D]
        pooled_patches = patch_tokens.mean(dim=1)  # [B, D]

        x = torch.cat([cls_token, pooled_patches], dim=1)  # [B, 2D]
        logits = self.classifier(x)
        return logits


@dataclass
class DinoClassifier:
    model: nn.Module
    labels: List[str]
    device: str
    transform: T.Compose

    @torch.inference_mode()
    def predict_proba(self, img: Image.Image) -> List[float]:
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        proba = torch.softmax(logits, dim=-1).detach().cpu().tolist()[0]
        return proba

def load_dino_classifier(
    checkpoint_path: Path,
    dinov3_repo_dir: Path,
    device: str,
) -> DinoClassifier:
    """
    Loads:
      - checkpoint with 'labels' + 'model_state_dict'
      - wraps with CLS+mean-pool head (DinoV3Head)
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    labels = list(checkpoint["labels"])
    num_classes = len(labels)

    backbone = torch.hub.load(
        str(dinov3_repo_dir),
        "dinov3_vitb16",
        source="local",
    ).to(device)

    model = DinoV3Head(backbone=backbone, num_classes=num_classes).to(device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return DinoClassifier(
        model=model,
        labels=labels,
        device=device,
        transform=build_val_transform(),
    )
