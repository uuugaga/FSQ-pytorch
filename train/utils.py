import yaml
import math
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If specified file doesn't exist
        ValueError: If YAML parsing fails
    """
    try:
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {yaml_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in {yaml_path}") from e


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup and decay phases.

    Supports multiple decay strategies:
    - Linear decay
    - Cosine decay
    - Exponential decay
    - No decay

    Attributes:
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate during warmup
        max_lr: Maximum learning rate after warmup
        decay_type: Type of decay strategy
        decay_ratio: Final learning rate ratio relative to max_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        max_lr: float = 3e-4,
        decay_type: str = "cosine",
        decay_ratio: float = 0.1,
    ):
        # Parameter validation
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps} (must be ≥0)")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) < warmup_steps ({warmup_steps})"
            )
        if min_lr < 0:
            raise ValueError(f"Invalid min_lr: {min_lr} (must be ≥0)")
        if max_lr <= min_lr:
            raise ValueError(f"max_lr ({max_lr}) must be > min_lr ({min_lr})")
        if decay_ratio < 0 or decay_ratio > 1:
            raise ValueError(f"decay_ratio must be in [0, 1], got {decay_ratio}")
        if decay_type not in ["linear", "cosine", "exponential", "none"]:
            raise ValueError(f"Invalid decay_type: {decay_type}")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_type = decay_type
        self.decay_ratio = decay_ratio
        self.final_lr = max_lr * decay_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        current_step = self.last_epoch + 1  # Adjust for 0-based indexing

        # Warmup phase
        if current_step <= self.warmup_steps:
            progress = current_step / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        # Decay phase
        else:
            decay_progress = current_step - self.warmup_steps
            decay_steps = self.total_steps - self.warmup_steps

            if self.decay_type == "none":
                lr = self.max_lr
            elif self.decay_type == "linear":
                lr = self._linear_decay(decay_progress, decay_steps)
            elif self.decay_type == "cosine":
                lr = self._cosine_decay(decay_progress, decay_steps)
            elif self.decay_type == "exponential":
                lr = self._exponential_decay(decay_progress, decay_steps)

        # Scale learning rates relative to original optimizer rates
        return [lr * (base / self.max_lr) for base in self.base_lrs]

    def _linear_decay(self, step: int, total_steps: int) -> float:
        """Linear decay computation."""
        progress = step / total_steps
        return self.max_lr - (self.max_lr - self.final_lr) * progress

    def _cosine_decay(self, step: int, total_steps: int) -> float:
        """Cosine decay computation."""
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        return self.final_lr + (self.max_lr - self.final_lr) * cosine_decay

    def _exponential_decay(self, step: int, total_steps: int) -> float:
        """Exponential decay computation."""
        decay_rate = (self.final_lr / self.max_lr) ** (1 / total_steps)
        return self.max_lr * (decay_rate**step)

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate for next epoch."""
        super().step(epoch)  # Handles epoch increment and logging

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "decay_type": self.decay_type,
            "decay_ratio": self.decay_ratio,
            "final_lr": self.final_lr,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.__dict__.update(state_dict)
        self._init_base_lrs()  # Ensures base_lrs are properly restored
