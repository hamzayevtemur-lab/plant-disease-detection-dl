"""
Experiment 1: Baseline CNN
--------------------------
Architecture:
- 4 Convolutional blocks (32 -> 64 -> 128 -> 256 channels)
- Kernel size: 3x3, Padding: 1, MaxPool: 2x2
- Global Adaptive AvgPool + Dropout(0.5) + Linear(256 -> 38 classes)
- Optimizer: Adam (lr=0.001)
- Epochs: 10, Batch size: 32
"""

import argparse
import sys
import torch

from src.models.cnn import PlantDiseaseCNN, CNNConfig
from src.training.trainer import TrainingConfig, run_experiment, get_device


def get_model_config() -> CNNConfig:
    return CNNConfig(
        name="baseline",
        num_classes=38,
        in_channels=3,
        channels=(32, 64, 128, 256),
        kernel_size=3,
        pool_size=2,
        dropout=0.5
    )


def get_training_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        epochs=10,
        optimizer="adam"
    )


def show_summary():
    model_config = get_model_config()
    training_config = get_training_config()
    model = PlantDiseaseCNN(model_config)

    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {model_config.name.upper()}")
    print("=" * 60)
    print("\n[1] Architecture Configuration:")
    print(model.summary())

    print("\n[2] Training Hyperparameters:")
    print(f"  - Optimizer:     {training_config.optimizer.upper()}")
    print(f"  - Learning Rate: {training_config.learning_rate}")
    print(f"  - Weight Decay:  {training_config.weight_decay}")
    print(f"  - Batch Size:    {training_config.batch_size}")
    print(f"  - Epochs:        {training_config.epochs}")

    # Forward pass dry-run check
    sample_input = torch.randn(2, 3, 224, 224)
    sample_output = model(sample_input)
    print("\n[3] Tensor Flow Verification:")
    print(f"  - Input shape:   {tuple(sample_input.shape)} (B, C, H, W)")
    print(f"  - Output shape:  {tuple(sample_output.shape)} (B, num_classes)")
    print("=" * 60 + "\n")


def train():
    model_config = get_model_config()
    training_config = get_training_config()
    show_summary()
    run_experiment(model_config=model_config, training_config=training_config)


def main():
    parser = argparse.ArgumentParser(description="Baseline CNN Experiment")
    parser.add_argument("--train", action="store_true", help="Run full training loop")
    parser.add_argument("--summary", action="store_true", help="Display model summary and structure")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        # Default behavior: show model summary and guide the user
        show_summary()
        print("💡 To train this model, run with '--train':")
        print(f"   python -m src.experiments.baseline --train\n")


if __name__ == "__main__":
    main()
