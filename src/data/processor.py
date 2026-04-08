"""Dataset loading and preprocessing utilities for SFT fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import datasets

from src.training.config import DataConfig

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = "あなたは誠実で優秀なアシスタントです。"


class DatasetProcessor:
    """Load and convert JSONL datasets into messages-format Datasets for SFTTrainer.

    Attributes:
        data_config: Dataset configuration including file paths and max_length.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def format_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a raw sample dict into messages format.

        Args:
            sample: Dict with ``instruction``, optionally ``input``, and ``output`` fields.

        Returns:
            ``{'messages': [...]}`` dict on success, or ``None`` if required fields are missing.
        """
        instruction: str | None = sample.get("instruction")
        output: str | None = sample.get("output")

        if not instruction or not output:
            return None

        input_text: str = sample.get("input") or ""
        user_content = instruction + ("\n" + input_text if input_text else "")

        messages = [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return {"messages": messages}

    def load_dataset(self, file_path: str) -> datasets.Dataset:
        """Load a JSONL file and return a Dataset with a ``messages`` column.

        Each line of the JSONL file is parsed as JSON and converted via
        :meth:`format_sample`. Lines that fail to parse are skipped with a
        warning; lines with missing required fields raise a warning and are
        also skipped.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            Dataset with a ``messages`` column.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping line %d in %s: JSON parse error — %s", lineno, file_path, exc)
                    continue

                formatted = self.format_sample(raw)
                if formatted is None:
                    logger.warning(
                        "Skipping line %d in %s: missing required fields ('instruction' or 'output').",
                        lineno,
                        file_path,
                    )
                    continue

                records.append(formatted)

        return datasets.Dataset.from_list(records)

    def load_and_process_dataset(self, file_path: str) -> datasets.Dataset:
        """Load a JSONL file and return a processed Dataset, skipping invalid samples.

        This is an alias for :meth:`load_dataset` that makes the processing
        intent explicit at call sites.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            Dataset containing only valid samples with a ``messages`` column.
        """
        return self.load_dataset(file_path)

    def get_train_eval_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load training and optional evaluation datasets.

        Returns:
            A tuple of ``(train_dataset, eval_dataset)``.  ``eval_dataset`` is
            ``None`` when :attr:`DataConfig.eval_file` is not configured.
        """
        train_dataset = self.load_and_process_dataset(self.data_config.train_file)

        eval_dataset: datasets.Dataset | None = None
        if self.data_config.eval_file is not None:
            eval_dataset = self.load_and_process_dataset(self.data_config.eval_file)

        return train_dataset, eval_dataset
