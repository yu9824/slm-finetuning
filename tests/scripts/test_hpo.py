"""tests/scripts/test_hpo.py: HPORunner のユニットテスト。

TDDフェーズ1: テスト先行（scripts/hpo.py の実装前に作成）。
モックを使用して Optuna / Ray Tune の実行を回避する。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートを sys.path に追加（conftest.py が存在するが念のため）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.hpo import HPORunner  # noqa: E402
from src.training.config import HPOConfig, LoRAConfig, TrainConfig  # noqa: E402


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config() -> TrainConfig:
    """テスト用のベース TrainConfig を返す。"""
    return TrainConfig(
        model_path="dummy/model",
        output_dir="outputs/base",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        lora=LoRAConfig(r=16, lora_alpha=32),
    )


@pytest.fixture()
def optuna_hpo_config() -> HPOConfig:
    """Optuna バックエンドの HPOConfig を返す。"""
    return HPOConfig(
        backend="optuna",
        n_trials=3,
        storage_path="hpo_results/test_optuna.db",
        best_params_file="hpo_results/test_best_params.yaml",
    )


@pytest.fixture()
def ray_hpo_config() -> HPOConfig:
    """Ray バックエンドの HPOConfig を返す。"""
    return HPOConfig(
        backend="ray",
        n_trials=3,
        storage_path="hpo_results/test_optuna.db",
        best_params_file="hpo_results/test_best_params.yaml",
    )


# ---------------------------------------------------------------------------
# テスト 1: run() が Optuna バックエンドを呼び出すことの確認
# ---------------------------------------------------------------------------


class TestHPORunnerOptunaBackend:
    """HPORunner.run() が Optuna バックエンドを正しく呼び出すことのテスト群。"""

    def test_run_calls_optuna_create_study(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """run() が optuna.create_study を呼び出すことを確認する。"""
        optuna_hpo_config.storage_path = str(tmp_path / "optuna.db")
        optuna_hpo_config.best_params_file = str(tmp_path / "best_params.yaml")

        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 5e-5, "lora_r": 8}
        mock_study.best_value = 0.42

        with patch("optuna.create_study", return_value=mock_study) as mock_create_study:
            runner = HPORunner(base_config, optuna_hpo_config)
            runner.run()

        mock_create_study.assert_called_once()
        call_kwargs = mock_create_study.call_args.kwargs
        assert call_kwargs.get("direction") == "minimize"
        assert call_kwargs.get("study_name") == "sarashina_hpo"

    def test_run_calls_study_optimize(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """run() が study.optimize を n_trials 回呼び出すことを確認する。"""
        optuna_hpo_config.storage_path = str(tmp_path / "optuna.db")
        optuna_hpo_config.best_params_file = str(tmp_path / "best_params.yaml")

        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 1e-4}
        mock_study.best_value = 0.35

        with patch("optuna.create_study", return_value=mock_study):
            runner = HPORunner(base_config, optuna_hpo_config)
            runner.run()

        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args.kwargs
        assert call_kwargs.get("n_trials") == optuna_hpo_config.n_trials

    def test_run_returns_best_params(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """run() が best_params の辞書を返すことを確認する。"""
        optuna_hpo_config.best_params_file = str(tmp_path / "best_params.yaml")
        optuna_hpo_config.storage_path = str(tmp_path / "optuna.db")

        expected_params = {"learning_rate": 2e-5, "lora_r": 16}
        mock_study = MagicMock()
        mock_study.best_params = expected_params
        mock_study.best_value = 0.25

        with patch("optuna.create_study", return_value=mock_study):
            runner = HPORunner(base_config, optuna_hpo_config)
            result = runner.run()

        assert result == expected_params


# ---------------------------------------------------------------------------
# テスト 2: run() が Ray バックエンドを呼び出すことの確認
# ---------------------------------------------------------------------------


class TestHPORunnerRayBackend:
    """HPORunner.run() が Ray バックエンドを正しく呼び出すことのテスト群。"""

    def test_run_calls_ray_init_and_tune_run(
        self,
        base_config: TrainConfig,
        ray_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """run() が ray.init と tune.run を呼び出すことを確認する。"""
        ray = pytest.importorskip("ray")

        ray_hpo_config.best_params_file = str(tmp_path / "best_params.yaml")
        ray_hpo_config.storage_path = str(tmp_path / "optuna.db")

        mock_best_config = {"learning_rate": 3e-5, "lora_r": 4}
        mock_analysis = MagicMock()
        mock_analysis.best_config = mock_best_config
        mock_analysis.best_result = {"eval_loss": 0.31}

        with (
            patch("ray.init") as mock_ray_init,
            patch("ray.tune.run", return_value=mock_analysis) as mock_tune_run,
            patch("ray.shutdown") as mock_ray_shutdown,
        ):
            runner = HPORunner(base_config, ray_hpo_config)
            result = runner.run()

        mock_ray_init.assert_called_once()
        mock_tune_run.assert_called_once()
        mock_ray_shutdown.assert_called_once()
        assert result == mock_best_config

    def test_run_ray_tune_run_receives_correct_num_samples(
        self,
        base_config: TrainConfig,
        ray_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """tune.run が num_samples=n_trials で呼び出されることを確認する。"""
        pytest.importorskip("ray")

        ray_hpo_config.best_params_file = str(tmp_path / "best_params.yaml")
        ray_hpo_config.storage_path = str(tmp_path / "optuna.db")

        mock_analysis = MagicMock()
        mock_analysis.best_config = {"learning_rate": 1e-4}
        mock_analysis.best_result = {"eval_loss": 0.50}

        with (
            patch("ray.init"),
            patch("ray.tune.run", return_value=mock_analysis) as mock_tune_run,
            patch("ray.shutdown"),
        ):
            runner = HPORunner(base_config, ray_hpo_config)
            runner.run()

        call_kwargs = mock_tune_run.call_args.kwargs
        assert call_kwargs.get("num_samples") == ray_hpo_config.n_trials
        assert call_kwargs.get("metric") == "eval_loss"
        assert call_kwargs.get("mode") == "min"


# ---------------------------------------------------------------------------
# テスト 3: _build_trial_config() が learning_rate を正しく適用することの確認
# ---------------------------------------------------------------------------


class TestBuildTrialConfig:
    """HPORunner._build_trial_config() のテスト群。"""

    def test_applies_learning_rate(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """learning_rate パラメータが trial_config に反映されることを確認する。"""
        runner = HPORunner(base_config, optuna_hpo_config)
        params = {"learning_rate": 5e-5, "trial_number": 0}
        trial_config = runner._build_trial_config(params)

        assert trial_config.learning_rate == 5e-5

    def test_applies_lora_r(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """lora_r パラメータが trial_config.lora.r に反映されることを確認する。"""
        runner = HPORunner(base_config, optuna_hpo_config)
        params = {"lora_r": 8, "trial_number": 1}
        trial_config = runner._build_trial_config(params)

        assert trial_config.lora.r == 8

    def test_applies_lora_alpha(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """lora_alpha パラメータが trial_config.lora.lora_alpha に反映されることを確認する。"""
        runner = HPORunner(base_config, optuna_hpo_config)
        params = {"lora_alpha": 64, "trial_number": 2}
        trial_config = runner._build_trial_config(params)

        assert trial_config.lora.lora_alpha == 64

    def test_applies_per_device_train_batch_size(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """per_device_train_batch_size が trial_config に反映されることを確認する。"""
        runner = HPORunner(base_config, optuna_hpo_config)
        params = {"per_device_train_batch_size": 4, "trial_number": 3}
        trial_config = runner._build_trial_config(params)

        assert trial_config.per_device_train_batch_size == 4

    def test_output_dir_contains_trial_number(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """output_dir にトライアル番号が含まれることを確認する。"""
        runner = HPORunner(base_config, optuna_hpo_config)
        params = {"trial_number": 7}
        trial_config = runner._build_trial_config(params)

        assert "trial_7" in trial_config.output_dir

    def test_does_not_mutate_base_config(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
    ) -> None:
        """_build_trial_config() が base_config を変更しないことを確認する。"""
        original_lr = base_config.learning_rate
        original_lora_r = base_config.lora.r

        runner = HPORunner(base_config, optuna_hpo_config)
        runner._build_trial_config({"learning_rate": 9e-6, "lora_r": 4, "trial_number": 0})

        assert base_config.learning_rate == original_lr
        assert base_config.lora.r == original_lora_r


# ---------------------------------------------------------------------------
# テスト 4: _save_best_params() が YAML ファイルを作成することの確認
# ---------------------------------------------------------------------------


class TestSaveBestParams:
    """HPORunner._save_best_params() のテスト群。"""

    def test_creates_yaml_file(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """_save_best_params() が YAML ファイルを作成することを確認する。"""
        import yaml

        output_file = tmp_path / "results" / "best_params.yaml"
        optuna_hpo_config.best_params_file = str(output_file)

        runner = HPORunner(base_config, optuna_hpo_config)
        best_params = {"learning_rate": 2e-5, "lora_r": 8}
        runner._save_best_params(best_params, best_value=0.15)

        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded["best_params"] == best_params
        assert loaded["best_eval_loss"] == pytest.approx(0.15)

    def test_creates_parent_directories(
        self,
        base_config: TrainConfig,
        optuna_hpo_config: HPOConfig,
        tmp_path: Path,
    ) -> None:
        """親ディレクトリが存在しない場合でも自動作成されることを確認する。"""
        output_file = tmp_path / "deep" / "nested" / "dir" / "best.yaml"
        optuna_hpo_config.best_params_file = str(output_file)

        runner = HPORunner(base_config, optuna_hpo_config)
        runner._save_best_params({"learning_rate": 1e-4}, best_value=0.99)

        assert output_file.exists()


# ---------------------------------------------------------------------------
# テスト 5: 不明な backend 指定時に ValueError が発生することの確認
# ---------------------------------------------------------------------------


class TestUnknownBackend:
    """不明な backend 指定時の挙動テスト群。"""

    def test_raises_value_error_for_unknown_backend(
        self,
        base_config: TrainConfig,
        tmp_path: Path,
    ) -> None:
        """backend が "optuna" でも "ray" でもない場合 ValueError が発生することを確認する。"""
        from typing import cast, Literal
        hpo_config = HPOConfig(
            backend=cast("Literal['optuna', 'ray']", "unknown_backend"),
            n_trials=1,
            best_params_file=str(tmp_path / "best.yaml"),
            storage_path=str(tmp_path / "optuna.db"),
        )

        runner = HPORunner(base_config, hpo_config)
        with pytest.raises(ValueError, match="Unknown backend"):
            runner.run()
