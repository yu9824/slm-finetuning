"""tests/scripts/test_inference.py: InferenceRunner のユニットテスト。

TDDフェーズ1: テスト先行（scripts/inference.py の実装前に作成）。
モックを使用してモデル・トークナイザーのロードを回避する。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートを sys.path に追加（conftest.py が存在するが念のため）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference import run_inference  # noqa: E402


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(encoded_input_ids: list[int] | None = None) -> MagicMock:
    """テスト用のモックトークナイザーを生成する。"""
    if encoded_input_ids is None:
        encoded_input_ids = [1, 2, 3]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<s>[INST] テスト [/INST]"

    # tokenizer(text, return_tensors="pt") の返り値
    encoded = MagicMock()
    input_ids_mock = MagicMock()
    input_ids_mock.__len__ = MagicMock(return_value=len(encoded_input_ids))
    # .shape[-1] で長さを取得するケースに対応
    input_ids_mock.shape = (1, len(encoded_input_ids))
    encoded.input_ids = input_ids_mock
    # .to(device) チェーン
    encoded.__getitem__ = MagicMock(return_value=encoded)
    mock_tokenizer.return_value = encoded

    # decode の返り値
    mock_tokenizer.decode.return_value = "生成されたテキスト"
    return mock_tokenizer


def _make_mock_model(generated_ids: list[int] | None = None) -> MagicMock:
    """テスト用のモックモデルを生成する。"""
    if generated_ids is None:
        generated_ids = [1, 2, 3, 4, 5]

    mock_model = MagicMock()
    output_ids = MagicMock()
    # output_ids[0] でトークン列を返す
    output_ids.__getitem__ = MagicMock(return_value=generated_ids)
    mock_model.generate.return_value = output_ids
    mock_model.device = "cpu"
    return mock_model


# ---------------------------------------------------------------------------
# テスト: 正常系
# ---------------------------------------------------------------------------


class TestRunInference:
    """run_inference() 関数のテスト群。"""

    @patch("scripts.inference.PeftModel")
    @patch("scripts.inference.AutoTokenizer")
    @patch("scripts.inference.AutoModelForCausalLM")
    def test_returns_string(
        self,
        mock_auto_model_cls: MagicMock,
        mock_auto_tokenizer_cls: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run_inference が文字列を返すことを確認する。"""
        base_model_path = str(tmp_path / "base_model")
        adapter_path = str(tmp_path / "adapter")
        Path(base_model_path).mkdir()
        Path(adapter_path).mkdir()

        mock_model = _make_mock_model()
        mock_auto_model_cls.from_pretrained.return_value = mock_model
        mock_peft_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = _make_mock_tokenizer()
        mock_auto_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        result = run_inference(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            prompt="テスト",
        )

        assert isinstance(result, str)

    @patch("scripts.inference.PeftModel")
    @patch("scripts.inference.AutoTokenizer")
    @patch("scripts.inference.AutoModelForCausalLM")
    def test_returns_decoded_text(
        self,
        mock_auto_model_cls: MagicMock,
        mock_auto_tokenizer_cls: MagicMock,
        mock_peft_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run_inference がデコードされた生成テキストを返すことを確認する。"""
        base_model_path = str(tmp_path / "base_model")
        adapter_path = str(tmp_path / "adapter")
        Path(base_model_path).mkdir()
        Path(adapter_path).mkdir()

        mock_model = _make_mock_model()
        mock_auto_model_cls.from_pretrained.return_value = mock_model
        mock_peft_cls.from_pretrained.return_value = mock_model

        expected_text = "期待されるレスポンス"
        mock_tokenizer = _make_mock_tokenizer()
        mock_tokenizer.decode.return_value = expected_text
        mock_auto_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        result = run_inference(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            prompt="テスト",
        )

        assert result == expected_text

    # ---------------------------------------------------------------------------
    # テスト: 異常系 - パス不在
    # ---------------------------------------------------------------------------

    def test_raises_file_not_found_when_base_model_path_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """base_model_path が存在しない場合 FileNotFoundError が発生する。"""
        missing_path = str(tmp_path / "nonexistent_model")
        adapter_path = str(tmp_path / "adapter")
        Path(adapter_path).mkdir()

        with pytest.raises(FileNotFoundError, match="base_model_path"):
            run_inference(
                base_model_path=missing_path,
                adapter_path=adapter_path,
                prompt="テスト",
            )

    def test_raises_file_not_found_when_adapter_path_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """adapter_path が存在しない場合 FileNotFoundError が発生する。"""
        base_model_path = str(tmp_path / "base_model")
        Path(base_model_path).mkdir()
        missing_adapter = str(tmp_path / "nonexistent_adapter")

        with pytest.raises(FileNotFoundError, match="adapter_path"):
            run_inference(
                base_model_path=base_model_path,
                adapter_path=missing_adapter,
                prompt="テスト",
            )


# ---------------------------------------------------------------------------
# テスト: main() / CLI 統合テスト
# ---------------------------------------------------------------------------


class TestMainEnforceOffline:
    """main() が enforce_offline を呼び出すことを確認する統合テスト群。"""

    @patch("scripts.inference.run_inference")
    @patch("scripts.inference.enforce_offline")
    def test_enforce_offline_called_from_main(
        self,
        mock_enforce_offline: MagicMock,
        mock_run_inference: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """main() が enforce_offline() を呼び出すことを確認する。"""
        from scripts.inference import main

        base_model_path = str(tmp_path / "base_model")
        adapter_path = str(tmp_path / "adapter")
        Path(base_model_path).mkdir()
        Path(adapter_path).mkdir()

        mock_run_inference.return_value = "生成テキスト"

        test_args = [
            "inference.py",
            "--base-model-path", base_model_path,
            "--adapter-path", adapter_path,
            "--prompt", "テスト",
        ]
        with patch("sys.argv", test_args):
            main()

        mock_enforce_offline.assert_called_once()

    @patch("scripts.inference.run_inference")
    @patch("scripts.inference.enforce_offline")
    def test_main_prints_generated_text(
        self,
        mock_enforce_offline: MagicMock,
        mock_run_inference: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """main() が生成テキストを標準出力に print することを確認する。"""
        from scripts.inference import main

        base_model_path = str(tmp_path / "base_model")
        adapter_path = str(tmp_path / "adapter")
        Path(base_model_path).mkdir()
        Path(adapter_path).mkdir()

        expected_output = "モックで生成されたテキスト"
        mock_run_inference.return_value = expected_output

        test_args = [
            "inference.py",
            "--base-model-path", base_model_path,
            "--adapter-path", adapter_path,
            "--prompt", "テスト",
        ]
        with patch("sys.argv", test_args):
            main()

        captured = capsys.readouterr()
        assert expected_output in captured.out

    @patch("scripts.inference.run_inference")
    @patch("scripts.inference.enforce_offline")
    def test_main_no_sample_flag(
        self,
        mock_enforce_offline: MagicMock,
        mock_run_inference: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--no-sample フラグで do_sample=False が渡されることを確認する。"""
        from scripts.inference import main

        base_model_path = str(tmp_path / "base_model")
        adapter_path = str(tmp_path / "adapter")
        Path(base_model_path).mkdir()
        Path(adapter_path).mkdir()

        mock_run_inference.return_value = "テキスト"

        test_args = [
            "inference.py",
            "--base-model-path", base_model_path,
            "--adapter-path", adapter_path,
            "--prompt", "テスト",
            "--no-sample",
        ]
        with patch("sys.argv", test_args):
            main()

        _, kwargs = mock_run_inference.call_args
        assert kwargs.get("do_sample") is False
