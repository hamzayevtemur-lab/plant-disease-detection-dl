from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


@dataclass
class TransferConfig:
    name: str = "resnet18"
    architecture: str = "resnet18"  # "resnet18", "resnet50", "mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0"
    num_classes: int = 38
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.3


class PlantDiseaseTransferModel(nn.Module):
    """Transfer learning wrapper around popular pretrained vision architectures."""
    def __init__(self, config: TransferConfig):
        super().__init__()
        self.config = config
        arch = config.architecture.lower()

        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes)
            )
            self.model = backbone

        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes)
            )
            self.model = backbone

        elif arch in ("mobilenet_v3", "mobilenet_v3_large"):
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if config.pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            in_features = backbone.classifier[0].out_features if hasattr(backbone.classifier, '__getitem__') else 960
            backbone.classifier = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(),
                nn.Dropout(config.dropout),
                nn.Linear(1280, config.num_classes)
            )
            self.model = backbone

        elif arch == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            backbone.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(config.dropout),
                nn.Linear(1024, config.num_classes)
            )
            self.model = backbone

        elif arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes)
            )
            self.model = backbone

        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")

        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freezes all layers except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier / fc
        classifier = getattr(self.model, "fc", None) or getattr(self.model, "classifier", None)
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            f"Model: {self.config.architecture.upper()} ({self.config.name})",
            f"Pretrained on ImageNet: {self.config.pretrained}",
            f"Freeze Backbone: {self.config.freeze_backbone}",
            f"Dropout: {self.config.dropout}",
            f"Output classes: {self.config.num_classes}",
            f"Total parameters: {self.count_parameters():,}",
            f"Trainable parameters: {self.count_trainable_parameters():,}"
        ]
        return "\n".join(lines)
