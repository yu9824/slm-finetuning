"""tests/scripts/test_train.py: scripts/train.py のユニットテスト。

TDD フェーズ1: テスト先行（scripts/train.py の実装前に作成）。
モックを使用してモデルロード・学習・ファイルI/O を回避する。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# プロジェクトルートを sys.path に追加（conftest.py があるが念のため）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_mock_config(output_dir: str = "/tmp/test_output") -> MagicMock:
    """テスト用モック TrainConfig を生成する。"""
    mock_config = MagicMock()
    mock_config.output_dir = output_dir
    mock_config.model_path = "/tmp/fake_model"
    mock_config.lora.use_qlora = False
    mock_config.gradient_checkpointing = False
    return mock_config


def _make_mock_trial_result(eval_loss: float = 0.1234) -> MagicMock:
    """テスト用モック TrialResult を生成する。"""
    mock_result = MagicMock()
    mock_result.eval_loss = eval_loss
    mock_result.model_path = "/tmp/test_output"
    return mock_result


# ---------------------------------------------------------------------------
# テスト: enforce_offline の呼び出し確認
# ---------------------------------------------------------------------------


class TestEnforceOfflineCalled:
    """main() が enforce_offline() を呼び出すことを確認するテスト群。"""

    @patch("scripts.train.PeftModel")
    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_enforce_offline_called(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_load_model: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """main() が enforce_offline() を呼び出すことを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_load_config.return_value = _make_mock_config(output_dir)
        mock_run_trial.return_value = _make_mock_trial_result()
        mock_load_model.return_value = (MagicMock(), MagicMock())

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        mock_enforce_offline.assert_called_once()


# ---------------------------------------------------------------------------
# テスト: run_training_trial の呼び出し確認
# ---------------------------------------------------------------------------


class TestRunTrainingTrialCalled:
    """main() が run_training_trial() を呼び出すことを確認するテスト群。"""

    @patch("scripts.train.PeftModel")
    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_run_training_trial_called(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_load_model: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """main() が run_training_trial() を呼び出し、config を渡すことを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_config = _make_mock_config(output_dir)
        mock_load_config.return_value = mock_config
        mock_run_trial.return_value = _make_mock_trial_result()
        mock_load_model.return_value = (MagicMock(), MagicMock())

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        mock_run_trial.assert_called_once_with(mock_config)

    @patch("scripts.train.PeftModel")
    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_main_prints_eval_loss(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_load_model: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """main() が eval_loss を標準出力に print することを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_load_config.return_value = _make_mock_config(output_dir)
        mock_run_trial.return_value = _make_mock_trial_result(eval_loss=0.5678)
        mock_load_model.return_value = (MagicMock(), MagicMock())

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "0.5678" in captured.out


# ---------------------------------------------------------------------------
# テスト: save_config_snapshot の呼び出し確認
# ---------------------------------------------------------------------------


class TestSaveConfigSnapshotCalled:
    """main() が save_config_snapshot() を呼び出すことを確認するテスト群。"""

    @patch("scripts.train.PeftModel")
    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_save_config_snapshot_called(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_load_model: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """main() が save_config_snapshot() を config と output_dir で呼び出すことを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_config = _make_mock_config(output_dir)
        mock_load_config.return_value = mock_config
        mock_run_trial.return_value = _make_mock_trial_result()
        mock_load_model.return_value = (MagicMock(), MagicMock())

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        mock_save_snapshot.assert_called_once_with(mock_config, output_dir)


# ---------------------------------------------------------------------------
# テスト: output_dir の自動作成確認
# ---------------------------------------------------------------------------


class TestOutputDirCreated:
    """output_dir が存在しない場合に自動作成されることを確認するテスト群。"""

    @patch("scripts.train.PeftModel")
    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_output_dir_is_created(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_load_model: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """main() が output_dir を自動作成することを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "new_nested" / "output")
        mock_load_config.return_value = _make_mock_config(output_dir)
        mock_run_trial.return_value = _make_mock_trial_result()
        mock_load_model.return_value = (MagicMock(), MagicMock())

        assert not Path(output_dir).exists()

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        assert Path(output_dir).exists()


# ---------------------------------------------------------------------------
# テスト: --merge フラグの動作確認
# ---------------------------------------------------------------------------


class TestMergeFlag:
    """--merge フラグが指定された場合の動作を確認するテスト群。"""

    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.PeftModel")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_merge_calls_peft_from_pretrained(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_peft_cls: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--merge フラグで PeftModel.from_pretrained() が呼び出されることを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_config = _make_mock_config(output_dir)
        mock_load_config.return_value = mock_config
        mock_run_trial.return_value = _make_mock_trial_result()

        mock_base_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_base_model, mock_tokenizer)

        mock_peft_model = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        test_args = ["train.py", "--output-dir", output_dir, "--merge"]
        with patch("sys.argv", test_args):
            main()

        # from_pretrained が output_dir（アダプターパス）で呼ばれていることを確認
        # （デモ推論でも呼ばれるため assert_any_call を使用）
        mock_peft_cls.from_pretrained.assert_any_call(mock_base_model, output_dir)

    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.PeftModel")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_merge_calls_merge_and_unload(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_peft_cls: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--merge フラグで merge_and_unload() が呼び出されることを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_config = _make_mock_config(output_dir)
        mock_load_config.return_value = mock_config
        mock_run_trial.return_value = _make_mock_trial_result()

        mock_base_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_base_model, mock_tokenizer)

        mock_peft_model = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        test_args = ["train.py", "--output-dir", output_dir, "--merge"]
        with patch("sys.argv", test_args):
            main()

        mock_peft_model.merge_and_unload.assert_called_once()

    @patch("scripts.train.load_model_and_tokenizer")
    @patch("scripts.train.PeftModel")
    @patch("scripts.train.run_training_trial")
    @patch("scripts.train.save_config_snapshot")
    @patch("scripts.train.load_config")
    @patch("scripts.train.enforce_offline")
    def test_merge_not_called_without_flag(
        self,
        mock_enforce_offline: MagicMock,
        mock_load_config: MagicMock,
        mock_save_snapshot: MagicMock,
        mock_run_trial: MagicMock,
        mock_peft_cls: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--merge フラグなしでは PeftModel.from_pretrained() が呼ばれないことを確認する。"""
        from scripts.train import main

        output_dir = str(tmp_path / "output")
        mock_config = _make_mock_config(output_dir)
        mock_load_config.return_value = mock_config
        mock_run_trial.return_value = _make_mock_trial_result()

        mock_base_model = MagicMock()
        mock_tokenizer = MagicMock()
        # merge なしでも demo inference で load_model_and_tokenizer は呼ばれるが
        # PeftModel.from_pretrained は merge 用には呼ばれない
        mock_load_model.return_value = (mock_base_model, mock_tokenizer)
        mock_peft_model = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        test_args = ["train.py", "--output-dir", output_dir]
        with patch("sys.argv", test_args):
            main()

        # merge 用の from_pretrained は呼ばれるが merge_and_unload は呼ばれない
        mock_peft_model.merge_and_unload.assert_not_called()
