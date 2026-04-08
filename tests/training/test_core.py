"""Tests for src.training.core.run_training_trial."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from src.training.config import DataConfig, LoRAConfig, TrainConfig
from src.training.core import TrialResult, run_training_trial


def _make_config(
    *,
    model_path: str = "/fake/model",
    output_dir: str = "/fake/output",
    eval_file: str | None = "data/eval.jsonl",
) -> TrainConfig:
    """テスト用 TrainConfig を生成するヘルパー。"""
    return TrainConfig(
        model_path=model_path,
        output_dir=output_dir,
        data=DataConfig(
            train_file="data/train.jsonl",
            eval_file=eval_file,
        ),
        lora=LoRAConfig(),
    )


# ---------------------------------------------------------------------------
# パッチ対象を一括管理
# ---------------------------------------------------------------------------
_PATCHES = {
    "load_model": "src.training.core.load_model_and_tokenizer",
    "apply_lora": "src.training.core.apply_lora",
    "DatasetProcessor": "src.training.core.DatasetProcessor",
    "SFTTrainer": "src.training.core.SFTTrainer",
    "SFTConfig": "src.training.core.SFTConfig",
}


def _build_mocks() -> dict[str, MagicMock]:
    """各パッチのデフォルトモックを構築して返す。"""
    mock_model = MagicMock(name="model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_lora_model = MagicMock(name="lora_model")
    mock_train_dataset = MagicMock(name="train_dataset")
    mock_eval_dataset = MagicMock(name="eval_dataset")
    mock_trainer = MagicMock(name="trainer")
    mock_trainer.evaluate.return_value = {"eval_loss": 0.42}

    mock_load = MagicMock(return_value=(mock_model, mock_tokenizer))
    mock_apply = MagicMock(return_value=mock_lora_model)

    mock_processor_instance = MagicMock(name="processor_instance")
    mock_processor_instance.get_train_eval_datasets.return_value = (
        mock_train_dataset,
        mock_eval_dataset,
    )
    mock_processor_cls = MagicMock(return_value=mock_processor_instance)

    mock_trainer_cls = MagicMock(return_value=mock_trainer)
    mock_sft_config_cls = MagicMock(return_value=MagicMock(name="sft_config"))

    return {
        "load_model": mock_load,
        "apply_lora": mock_apply,
        "DatasetProcessor": mock_processor_cls,
        "SFTTrainer": mock_trainer_cls,
        "SFTConfig": mock_sft_config_cls,
        # 便宜上 trainer オブジェクト自体も保持
        "_trainer": mock_trainer,
    }


class TestRunTrainingTrialReturnsTrialResult:
    """run_training_trial が TrialResult を返すことを検証する。"""

    def test_returns_trial_result_instance(self) -> None:
        """戻り値が TrialResult であることを確認する。"""
        config = _make_config()
        mocks = _build_mocks()

        with (
            patch(_PATCHES["load_model"], mocks["load_model"]),
            patch(_PATCHES["apply_lora"], mocks["apply_lora"]),
            patch(_PATCHES["DatasetProcessor"], mocks["DatasetProcessor"]),
            patch(_PATCHES["SFTTrainer"], mocks["SFTTrainer"]),
            patch(_PATCHES["SFTConfig"], mocks["SFTConfig"]),
        ):
            result = run_training_trial(config)

        assert isinstance(result, TrialResult)


class TestRunTrainingTrialEvalLoss:
    """eval_loss が trainer.evaluate() の結果から取得されることを検証する。"""

    def test_eval_loss_from_trainer_evaluate(self) -> None:
        """eval_loss が trainer.evaluate() の返り値と一致することを確認する。"""
        config = _make_config()
        mocks = _build_mocks()
        expected_loss = 0.42
        mocks["_trainer"].evaluate.return_value = {"eval_loss": expected_loss}

        with (
            patch(_PATCHES["load_model"], mocks["load_model"]),
            patch(_PATCHES["apply_lora"], mocks["apply_lora"]),
            patch(_PATCHES["DatasetProcessor"], mocks["DatasetProcessor"]),
            patch(_PATCHES["SFTTrainer"], mocks["SFTTrainer"]),
            patch(_PATCHES["SFTConfig"], mocks["SFTConfig"]),
        ):
            result = run_training_trial(config)

        assert result.eval_loss == pytest.approx(expected_loss)

    def test_eval_loss_is_nan_when_no_eval_dataset(self) -> None:
        """eval_file が None の場合、eval_loss が NaN であることを確認する。"""
        config = _make_config(eval_file=None)
        mocks = _build_mocks()

        # eval_dataset を None に設定
        mock_processor_instance = mocks["DatasetProcessor"].return_value
        mock_processor_instance.get_train_eval_datasets.return_value = (
            MagicMock(name="train_dataset"),
            None,
        )

        with (
            patch(_PATCHES["load_model"], mocks["load_model"]),
            patch(_PATCHES["apply_lora"], mocks["apply_lora"]),
            patch(_PATCHES["DatasetProcessor"], mocks["DatasetProcessor"]),
            patch(_PATCHES["SFTTrainer"], mocks["SFTTrainer"]),
            patch(_PATCHES["SFTConfig"], mocks["SFTConfig"]),
        ):
            result = run_training_trial(config)

        assert math.isnan(result.eval_loss)


class TestRunTrainingTrialModelPath:
    """model_path が config.output_dir であることを検証する。"""

    def test_model_path_equals_output_dir(self) -> None:
        """TrialResult.model_path が config.output_dir と一致することを確認する。"""
        expected_output_dir = "/some/custom/output"
        config = _make_config(output_dir=expected_output_dir)
        mocks = _build_mocks()

        with (
            patch(_PATCHES["load_model"], mocks["load_model"]),
            patch(_PATCHES["apply_lora"], mocks["apply_lora"]),
            patch(_PATCHES["DatasetProcessor"], mocks["DatasetProcessor"]),
            patch(_PATCHES["SFTTrainer"], mocks["SFTTrainer"]),
            patch(_PATCHES["SFTConfig"], mocks["SFTConfig"]),
        ):
            result = run_training_trial(config)

        assert result.model_path == expected_output_dir


class TestRunTrainingTrialFileNotFound:
    """model_path が存在しない場合に FileNotFoundError が発生することを検証する。"""

    def test_raises_file_not_found_error_when_model_path_missing(self) -> None:
        """load_model_and_tokenizer が FileNotFoundError を送出した場合に再送出されることを確認する。"""
        config = _make_config(model_path="/nonexistent/path")
        mocks = _build_mocks()
        mocks["load_model"].side_effect = FileNotFoundError(
            "Model path does not exist: /nonexistent/path"
        )

        with (
            patch(_PATCHES["load_model"], mocks["load_model"]),
            patch(_PATCHES["apply_lora"], mocks["apply_lora"]),
            patch(_PATCHES["DatasetProcessor"], mocks["DatasetProcessor"]),
            patch(_PATCHES["SFTTrainer"], mocks["SFTTrainer"]),
            patch(_PATCHES["SFTConfig"], mocks["SFTConfig"]),
        ):
            with pytest.raises(FileNotFoundError):
                run_training_trial(config)
