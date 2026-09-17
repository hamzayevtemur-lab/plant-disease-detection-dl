# Re-export TrainingConfig from trainer.py for backwards compatibility with saved torch checkpoints
from src.training.trainer import TrainingConfig

__all__ = ["TrainingConfig"]
