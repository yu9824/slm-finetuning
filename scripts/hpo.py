"""scripts/hpo.py: ハイパーパラメータ最適化 (HPO) スクリプト。

Optuna または Ray Tune を使用してハイパーパラメータを自動探索し、
最良設定を YAML ファイルに出力する。
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import yaml

try:
    import ray
    from ray import tune

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

from src.training.config import HPOConfig, TrainConfig, load_config
from src.training.core import run_training_trial
from src.utils.offline import enforce_offline

logger = logging.getLogger(__name__)


class HPORunner:
    """ハイパーパラメータ最適化を実行するクラス。

    Optuna または Ray Tune バックエンドを使用して探索空間を走査し、
    最小 eval_loss を達成するハイパーパラメータを探索する。

    Attributes:
        base_config: ベースとなる TrainConfig。探索パラメータで上書きしてコピーを作成する。
        hpo_config: HPO の実行設定（バックエンド、試行数、出力先など）。
    """

    def __init__(self, base_config: TrainConfig, hpo_config: HPOConfig) -> None:
        """HPORunner を初期化する。

        Args:
            base_config: ベースとなる TrainConfig。
            hpo_config: HPO 実行設定。
        """
        self.base_config = base_config
        self.hpo_config = hpo_config

    def run(self) -> dict[str, Any]:
        """HPO を実行し、最良パラメータを返す。

        Returns:
            最良ハイパーパラメータの辞書。

        Raises:
            ValueError: backend が "optuna" でも "ray" でもない場合。
            ImportError: backend="ray" を指定したが ray がインストールされていない場合。
        """
        if self.hpo_config.backend == "optuna":
            best_params, best_value = self._run_optuna()
        elif self.hpo_config.backend == "ray":
            if not _RAY_AVAILABLE:
                raise ImportError(
                    "ray はインストールされていません。`pip install ray[tune]` を実行してください。"
                )
            best_params, best_value = self._run_ray()
        else:
            raise ValueError(
                f"Unknown backend: {self.hpo_config.backend!r}. "
                "'optuna' または 'ray' を指定してください。"
            )

        self._save_best_params(best_params, best_value)
        return best_params

    def _optuna_objective(self, trial: Any) -> float:
        """Optuna の目的関数。eval_loss を最小化する。

        Args:
            trial: Optuna の Trial オブジェクト。

        Returns:
            eval_loss（最小化目標）。
        """
        params: dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64]),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [1, 2, 4]
            ),
            "trial_number": trial.number,
        }
        trial_config = self._build_trial_config(params)
        result = run_training_trial(trial_config)
        return result.eval_loss

    def _build_trial_config(self, params: dict[str, Any]) -> TrainConfig:
        """探索パラメータで TrainConfig を更新したコピーを作成する。

        base_config は変更しない（deepcopy を使用）。
        output_dir にはトライアル番号を含め、複数トライアル間の衝突を防ぐ。

        Args:
            params: 適用するハイパーパラメータの辞書。
                "learning_rate", "lora_r", "lora_alpha",
                "per_device_train_batch_size", "trial_number" のキーを持つ。

        Returns:
            パラメータを適用した TrainConfig のコピー。
        """
        trial_config = copy.deepcopy(self.base_config)

        trial_config.learning_rate = params.get(
            "learning_rate", self.base_config.learning_rate
        )
        trial_config.per_device_train_batch_size = params.get(
            "per_device_train_batch_size", self.base_config.per_device_train_batch_size
        )
        trial_config.lora.r = params.get("lora_r", self.base_config.lora.r)
        trial_config.lora.lora_alpha = params.get(
            "lora_alpha", self.base_config.lora.lora_alpha
        )

        trial_number = params.get("trial_number", 0)
        trial_config.output_dir = str(
            Path(self.base_config.output_dir) / f"trial_{trial_number}"
        )

        return trial_config

    def _run_optuna(self) -> tuple[dict[str, Any], float]:
        """Optuna バックエンドで HPO を実行する。

        Returns:
            (best_params, best_value) のタプル。
        """
        import optuna

        storage_url = f"sqlite:///{self.hpo_config.storage_path}"
        Path(self.hpo_config.storage_path).parent.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            direction="minimize",
            storage=storage_url,
            study_name="sarashina_hpo",
            load_if_exists=True,
        )
        study.optimize(self._optuna_objective, n_trials=self.hpo_config.n_trials)

        return study.best_params, study.best_value

    def _run_ray(self) -> tuple[dict[str, Any], float]:
        """Ray Tune バックエンドで HPO を実行する。

        Returns:
            (best_params, best_value) のタプル。
        """
        ray.init(local_mode=True, ignore_reinit_error=True)

        search_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "lora_r": tune.choice([4, 8, 16, 32]),
            "lora_alpha": tune.choice([16, 32, 64]),
            "per_device_train_batch_size": tune.choice([1, 2, 4]),
        }

        # self を参照するためにクロージャで包む
        runner = self

        def ray_objective(params: dict[str, Any]) -> None:
            trial_config = runner._build_trial_config(params)
            result = run_training_trial(trial_config)
            tune.report({"eval_loss": result.eval_loss})

        analysis = tune.run(
            ray_objective,
            config=search_space,
            num_samples=self.hpo_config.n_trials,
            metric="eval_loss",
            mode="min",
        )
        best_params: dict[str, Any] = analysis.best_config
        best_value: float = analysis.best_result["eval_loss"]

        ray.shutdown()
        return best_params, best_value

    def _save_best_params(
        self, best_params: dict[str, Any], best_value: float
    ) -> None:
        """最良パラメータを YAML ファイルに保存する。

        Args:
            best_params: 最良ハイパーパラメータの辞書。
            best_value: 最良 eval_loss の値。
        """
        output_path = Path(self.hpo_config.best_params_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"best_params": best_params, "best_eval_loss": best_value},
                f,
                allow_unicode=True,
            )

        logger.info("最良パラメータを保存しました: %s", output_path)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        解析済み引数の Namespace。
    """
    parser = argparse.ArgumentParser(
        description="ハイパーパラメータ最適化スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 設定ファイルのパス",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="トライアル数",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["optuna", "ray"],
        default=None,
        help='HPO バックエンド ("optuna" または "ray")',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ベースディレクトリ",
    )
    return parser.parse_args()


def main() -> None:
    """HPO スクリプトのエントリーポイント。"""
    enforce_offline()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    cli_overrides: dict[str, Any] = {}
    if args.output_dir is not None:
        cli_overrides["output_dir"] = args.output_dir

    base_config = load_config(args.config, cli_overrides)

    hpo_config = HPOConfig()
    if args.n_trials is not None:
        hpo_config.n_trials = args.n_trials
    if args.backend is not None:
        hpo_config.backend = args.backend  # type: ignore[assignment]

    runner = HPORunner(base_config, hpo_config)
    best_params = runner.run()

    logger.info("HPO 完了。最良パラメータ: %s", best_params)
    print(f"最良パラメータ: {best_params}")


if __name__ == "__main__":
    main()
