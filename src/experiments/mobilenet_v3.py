"""
Experiment 5: MobileNetV3-Large (Production Edge Model)
-------------------------------------------------------
Architecture:
- MobileNetV3-Large pretrained on ImageNet-1K (optimized for low-latency & mobile)
- Custom classification head: Linear(960 -> 1280) -> Hardswish -> Dropout(0.2) -> Linear(1280 -> 38 classes)
- Optimizer: AdamW (lr=5e-4, weight_decay=1e-4) with Cosine Annealing LR Schedule
- Epochs: 10, Batch size: 32
"""

import argparse
import sys
import torch

from src.models.transfer import PlantDiseaseTransferModel, TransferConfig
from src.training.trainer import TrainingConfig, run_experiment, get_device


def get_model_config() -> TransferConfig:
    return TransferConfig(
        name="mobilenet_v3",
        architecture="mobilenet_v3_large",
        num_classes=38,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2
    )


def get_training_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-4,
        epochs=10,
        optimizer="adamw",
        use_lr_scheduler=True
    )


def show_summary():
    model_config = get_model_config()
    training_config = get_training_config()
    model = PlantDiseaseTransferModel(model_config)

    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {model_config.name.upper()} (MOBILE EDGE PRODUCTION)")
    print("=" * 60)
    print("\n[1] Architecture Configuration:")
    print(model.summary())

    print("\n[2] Training Hyperparameters:")
    print(f"  - Optimizer:       {training_config.optimizer.upper()}")
    print(f"  - Learning Rate:   {training_config.learning_rate}")
    print(f"  - Weight Decay:    {training_config.weight_decay}")
    print(f"  - LR Scheduler:    {'CosineAnnealingLR' if training_config.use_lr_scheduler else 'None'}")
    print(f"  - Batch Size:      {training_config.batch_size}")
    print(f"  - Epochs:          {training_config.epochs}")

    # Forward pass dry-run check
    sample_input = torch.randn(2, 3, 224, 224)
    sample_output = model(sample_input)
    print("\n[3] Tensor Flow Verification:")
    print(f"  - Input shape:     {tuple(sample_input.shape)} (B, C, H, W)")
    print(f"  - Output shape:    {tuple(sample_output.shape)} (B, num_classes)")
    print("=" * 60 + "\n")


def train():
    model_config = get_model_config()
    training_config = get_training_config()
    show_summary()
    run_experiment(model_config=model_config, training_config=training_config)


def main():
    parser = argparse.ArgumentParser(description="MobileNetV3 Transfer Learning Experiment")
    parser.add_argument("--train", action="store_true", help="Run full training loop")
    parser.add_argument("--summary", action="store_true", help="Display model summary and structure")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        show_summary()
        print("💡 To train this model, run with '--train':")
        print(f"   python -m src.experiments.mobilenet_v3 --train\n")


if __name__ == "__main__":
    main()
