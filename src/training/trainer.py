import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.cnn import PlantDiseaseCNN, CNNConfig
from src.models.transfer import PlantDiseaseTransferModel, TransferConfig


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    use_lr_scheduler: bool = True
    splits_dir: str = "data/splits"
    dataset_dir: str = "data/raw/plantvillage"


def get_device() -> torch.device:
    """Detects available hardware accelerator (MPS for Apple Silicon, CUDA for Nvidia, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_config: Union[CNNConfig, TransferConfig]) -> nn.Module:
    """Factory function to build appropriate model instance based on configuration."""
    if isinstance(model_config, CNNConfig) or getattr(model_config, "channels", None) is not None:
        return PlantDiseaseCNN(model_config)
    elif isinstance(model_config, TransferConfig) or getattr(model_config, "architecture", None) is not None:
        return PlantDiseaseTransferModel(model_config)
    else:
        raise ValueError(f"Unrecognized model configuration type: {type(model_config)}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_accuracy = correct / max(total, 1)
    return epoch_loss, epoch_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on a dataloader and returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_accuracy = correct / max(total, 1)
    return epoch_loss, epoch_accuracy


def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    opt_name = config.optimizer.lower()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif opt_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def run_experiment(
    model_config: Union[CNNConfig, TransferConfig],
    training_config: TrainingConfig,
    experiments_base_dir: str = "experiments",
    model: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """Trains any model (Custom CNN, ResNet/MobileNet), saves checkpoints, and evaluates test accuracy."""
    device = get_device()
    experiment_dir = os.path.join(experiments_base_dir, model_config.name)
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    results_path = os.path.join(experiment_dir, "results.json")

    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Starting Experiment: {model_config.name}")
    print(f"Hardware Device: {device}")
    print(f"{'=' * 60}")

    train_loader, validation_loader, test_loader = create_dataloaders(
        splits_dir=training_config.splits_dir,
        dataset_dir=training_config.dataset_dir,
        batch_size=training_config.batch_size
    )

    if model is None:
        model = build_model(model_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Total Parameters:     {total_params:,}")
    print(f"Model Trainable Parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, training_config)

    scheduler = None
    if training_config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.epochs,
            eta_min=1e-6
        )

    best_val_acc = -1.0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    for epoch in range(training_config.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{training_config.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "training_config": training_config,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "history": history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Best model checkpoint saved (Val Acc: {val_acc:.2%})")

    # Evaluate best checkpoint on test set
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"\nFinal Test Results for {model_config.name}:")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")

    results = {
        "experiment": model_config.name,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_epoch": checkpoint["epoch"],
        "validation_loss": checkpoint["validation_loss"],
        "validation_accuracy": checkpoint["validation_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": history
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_path}\n")

    return results


def evaluate_saved_model(
    experiment_name: str,
    experiments_base_dir: str = "experiments",
    splits_dir: str = "data/splits",
    dataset_dir: str = "data/raw/plantvillage"
) -> Dict[str, Any]:
    """Loads an existing best_model.pt checkpoint, evaluates on test data, and writes results.json."""
    device = get_device()
    experiment_dir = os.path.join(experiments_base_dir, experiment_name)
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    results_path = os.path.join(experiment_dir, "results.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    training_config = checkpoint["training_config"]

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader = create_dataloaders(
        splits_dir=splits_dir,
        dataset_dir=dataset_dir,
        batch_size=training_config.batch_size
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "experiment": experiment_name,
        "parameters": total_params,
        "best_epoch": checkpoint.get("epoch", 0),
        "validation_loss": checkpoint.get("validation_loss", 0.0),
        "validation_accuracy": checkpoint.get("validation_accuracy", 0.0),
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
    if "history" in checkpoint:
        results["history"] = checkpoint["history"]

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluated {experiment_name}: Test Acc = {test_acc:.2%}, Test Loss = {test_loss:.4f}")
    return results
