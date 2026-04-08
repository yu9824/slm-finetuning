"""Training configuration dataclasses and config management utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import yaml


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability applied to LoRA layers.
        target_modules: List of module names to apply LoRA to.
        bias: Bias training mode.
        use_qlora: Whether to use QLoRA (4-bit quantization).
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: Literal["none", "all", "lora_only"] = "none"
    use_qlora: bool = False


@dataclass
class DataConfig:
    """Dataset configuration.

    Attributes:
        train_file: Path to the training JSONL file.
        eval_file: Path to the evaluation JSONL file.
        max_length: Maximum token sequence length.
    """

    train_file: str = "data/dummy_train.jsonl"
    eval_file: Optional[str] = "data/dummy_eval.jsonl"
    max_length: int = 512


@dataclass
class TrainConfig:
    """Top-level training configuration.

    Attributes:
        model_path: Path to the base model (required).
        output_dir: Directory where checkpoints are saved (required).
        learning_rate: Optimizer learning rate.
        num_train_epochs: Total number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        gradient_checkpointing: Whether to use gradient checkpointing.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Run evaluation every N steps.
        report_to: List of experiment tracking backends.
        logging_dir: TensorBoard log directory.
        lora: LoRA adapter configuration.
        data: Dataset configuration.
    """

    model_path: str = field(default="")
    output_dir: str = field(default="")
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: str = "runs/"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration.

    Attributes:
        backend: HPO framework to use.
        n_trials: Number of HPO trials.
        storage_path: Optuna storage DB path.
        best_params_file: Output path for best hyperparameters.
        search_space: Search space definition for each hyperparameter.
    """

    backend: Literal["optuna", "ray"] = "optuna"
    n_trials: int = 10
    storage_path: str = "hpo_results/optuna.db"
    best_params_file: str = "hpo_results/best_params.yaml"
    search_space: dict[str, Any] = field(default_factory=dict)


def _apply_nested_overrides(config: TrainConfig, overrides: dict[str, Any]) -> None:
    """Apply flat or dot-notation overrides onto a TrainConfig in-place.

    Flat keys (e.g. ``learning_rate``) are applied directly to *config*.
    Dot-notation keys (e.g. ``lora.r``) resolve to the corresponding nested
    dataclass attribute.

    Args:
        config: The TrainConfig instance to mutate.
        overrides: Mapping of key -> value pairs to apply.
    """
    nested_map: dict[str, Any] = {
        "lora": config.lora,
        "data": config.data,
    }

    for key, value in overrides.items():
        if value is None:
            continue

        if "." in key:
            prefix, attr = key.split(".", maxsplit=1)
            target = nested_map.get(prefix)
            if target is not None and hasattr(target, attr):
                setattr(target, attr, value)
            else:
                raise ValueError(
                    f"Unknown override key '{key}': '{prefix}' has no attribute '{attr}'."
                )
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown override key '{key}' for TrainConfig.")


def _dict_to_train_config(raw: dict[str, Any]) -> TrainConfig:
    """Convert a raw dict (e.g. from YAML) into a TrainConfig.

    Nested keys ``lora`` and ``data`` are recursively converted into
    the corresponding dataclass instances.

    Args:
        raw: Dictionary representation of the configuration.

    Returns:
        Populated TrainConfig instance.
    """
    lora_raw: dict[str, Any] = raw.pop("lora", {}) or {}
    data_raw: dict[str, Any] = raw.pop("data", {}) or {}

    lora_config = LoRAConfig(**{k: v for k, v in lora_raw.items() if hasattr(LoRAConfig, k) or k in LoRAConfig.__dataclass_fields__})
    data_config = DataConfig(**{k: v for k, v in data_raw.items() if k in DataConfig.__dataclass_fields__})

    # Filter only valid TrainConfig fields from the remaining raw dict.
    valid_keys = TrainConfig.__dataclass_fields__.keys()
    top_level = {k: v for k, v in raw.items() if k in valid_keys}

    return TrainConfig(lora=lora_config, data=data_config, **top_level)


def load_config(
    config_path: Optional[str],
    cli_overrides: dict[str, Any],
) -> TrainConfig:
    """Load a TrainConfig from a YAML file and apply CLI overrides.

    When *config_path* is ``None`` or the file does not exist, a default
    TrainConfig is created from dataclass defaults before applying overrides.

    CLI overrides take precedence over file values.  Both flat keys
    (``learning_rate``) and dot-notation keys (``lora.r``) are supported.

    Args:
        config_path: Path to the YAML configuration file, or ``None``.
        cli_overrides: Mapping of override key-value pairs from the CLI.

    Returns:
        Fully resolved TrainConfig instance.

    Raises:
        ValueError: If an override key does not correspond to a known field.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        config = _dict_to_train_config(raw)
    else:
        config = TrainConfig()

    _apply_nested_overrides(config, cli_overrides)
    return config


def save_config_snapshot(config: TrainConfig, output_dir: str) -> None:
    """Persist the resolved configuration to *output_dir*/config_snapshot.yaml.

    The snapshot captures the exact configuration used for an experiment,
    including nested LoRA and data settings, to ensure reproducibility.

    Args:
        config: The TrainConfig instance to serialise.
        output_dir: Directory where ``config_snapshot.yaml`` is written.
            Created automatically if it does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")

    def _dataclass_to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _dataclass_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, list):
            return [_dataclass_to_dict(item) for item in obj]
        return obj

    snapshot = _dataclass_to_dict(config)

    with open(snapshot_path, "w", encoding="utf-8") as fh:
        yaml.dump(snapshot, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
