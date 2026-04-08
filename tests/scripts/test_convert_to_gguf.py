"""Tests for scripts.convert_to_gguf.GGUFConverter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# プロジェクトルートを sys.path に追加（conftest.py でも設定済みだが明示的に保証）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.convert_to_gguf import GGUFConverter


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_converter(
    tmp_path: Path,
    *,
    create_llama_cpp: bool = True,
    create_convert_script: bool = True,
    create_model: bool = True,
) -> GGUFConverter:
    """テスト用 GGUFConverter を構築するヘルパー。"""
    llama_cpp_path = tmp_path / "llama.cpp"
    model_path = tmp_path / "merged_model"
    output_dir = tmp_path / "output"

    if create_llama_cpp:
        llama_cpp_path.mkdir()
    if create_convert_script:
        (llama_cpp_path / "convert_hf_to_gguf.py").touch()
    if create_model:
        model_path.mkdir()

    return GGUFConverter(
        llama_cpp_path=str(llama_cpp_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
    )


# ---------------------------------------------------------------------------
# __init__ の検証
# ---------------------------------------------------------------------------


class TestGGUFConverterInit:
    """GGUFConverter.__init__ の異常系を検証する。"""

    def test_raises_file_not_found_when_llama_cpp_missing(
        self, tmp_path: Path
    ) -> None:
        """llama_cpp_path が存在しない場合に FileNotFoundError が発生することを確認する。"""
        with pytest.raises(FileNotFoundError, match="llama.cpp not found"):
            GGUFConverter(
                llama_cpp_path=str(tmp_path / "nonexistent"),
                model_path=str(tmp_path / "model"),
                output_dir=str(tmp_path / "output"),
            )

    def test_raises_file_not_found_when_convert_script_missing(
        self, tmp_path: Path
    ) -> None:
        """convert_hf_to_gguf.py が存在しない場合に FileNotFoundError が発生することを確認する。"""
        llama_cpp_path = tmp_path / "llama.cpp"
        llama_cpp_path.mkdir()
        # convert_hf_to_gguf.py を作成しない

        with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
            GGUFConverter(
                llama_cpp_path=str(llama_cpp_path),
                model_path=str(tmp_path / "model"),
                output_dir=str(tmp_path / "output"),
            )

    def test_error_message_contains_git_clone_hint(self, tmp_path: Path) -> None:
        """FileNotFoundError メッセージに git clone 手順が含まれることを確認する。"""
        with pytest.raises(FileNotFoundError, match="git clone"):
            GGUFConverter(
                llama_cpp_path=str(tmp_path / "nonexistent"),
                model_path=str(tmp_path / "model"),
                output_dir=str(tmp_path / "output"),
            )


# ---------------------------------------------------------------------------
# convert() の検証
# ---------------------------------------------------------------------------


class TestGGUFConverterConvert:
    """GGUFConverter.convert() の正常系・異常系を検証する。"""

    def test_convert_calls_subprocess_run(self, tmp_path: Path) -> None:
        """convert() が subprocess.run を呼び出すことを確認する。"""
        converter = _make_converter(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            converter.convert(quant_type="q4_k_m")

        assert mock_run.called

    def test_convert_f16_skips_quantize_step(self, tmp_path: Path) -> None:
        """quant_type='f16' の場合、量子化ステップをスキップすることを確認する。"""
        converter = _make_converter(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = converter.convert(quant_type="f16")

        # f16 変換のみ（llama-quantize は呼ばれない）
        assert mock_run.call_count == 1
        assert result.name == "model_f16.gguf"

    def test_convert_non_f16_calls_quantize(self, tmp_path: Path) -> None:
        """quant_type が f16 以外の場合、量子化ステップも呼び出すことを確認する。"""
        converter = _make_converter(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = converter.convert(quant_type="q4_k_m")

        # f16 変換 + 量子化の 2 回
        assert mock_run.call_count == 2
        assert result.name == "model_q4_k_m.gguf"

    def test_convert_quantize_uses_uppercase_quant_type(self, tmp_path: Path) -> None:
        """量子化コマンドで quant_type が大文字変換されることを確認する。"""
        converter = _make_converter(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            converter.convert(quant_type="q4_k_m")

        # 2 回目のコール（量子化）引数を取得
        quant_call_args = mock_run.call_args_list[1]
        cmd = quant_call_args[0][0]  # positional 第一引数がコマンドリスト
        assert "Q4_K_M" in cmd

    def test_convert_creates_output_dir(self, tmp_path: Path) -> None:
        """convert() が output_dir を自動作成することを確認する。"""
        converter = _make_converter(tmp_path)
        output_dir = Path(converter.output_dir)

        assert not output_dir.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            converter.convert(quant_type="f16")

        assert output_dir.exists()

    def test_convert_raises_runtime_error_on_subprocess_failure(
        self, tmp_path: Path
    ) -> None:
        """subprocess.run 失敗時に RuntimeError が発生することを確認する。"""
        converter = _make_converter(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["python", "convert_hf_to_gguf.py"]
            )
            with pytest.raises(RuntimeError):
                converter.convert(quant_type="q4_k_m")


# ---------------------------------------------------------------------------
# generate_modelfile() の検証
# ---------------------------------------------------------------------------


class TestGGUFConverterGenerateModelfile:
    """GGUFConverter.generate_modelfile() の正常系を検証する。"""

    def test_generate_modelfile_creates_file(self, tmp_path: Path) -> None:
        """generate_modelfile() が Modelfile を生成することを確認する。"""
        converter = _make_converter(tmp_path)
        output_dir = Path(converter.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = output_dir / "model_q4_k_m.gguf"
        gguf_path.touch()

        modelfile_path = converter.generate_modelfile(gguf_path, model_name="sarashina-ft")

        assert modelfile_path.exists()
        assert modelfile_path.name == "Modelfile"

    def test_generate_modelfile_contains_from_directive(self, tmp_path: Path) -> None:
        """Modelfile に FROM ディレクティブが含まれることを確認する。"""
        converter = _make_converter(tmp_path)
        output_dir = Path(converter.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = output_dir / "model_q4_k_m.gguf"
        gguf_path.touch()

        modelfile_path = converter.generate_modelfile(gguf_path)
        content = modelfile_path.read_text()

        assert "FROM ./model_q4_k_m.gguf" in content

    def test_generate_modelfile_contains_stop_parameters(self, tmp_path: Path) -> None:
        """Modelfile に PARAMETER stop が含まれることを確認する。"""
        converter = _make_converter(tmp_path)
        output_dir = Path(converter.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = output_dir / "model_f16.gguf"
        gguf_path.touch()

        modelfile_path = converter.generate_modelfile(gguf_path)
        content = modelfile_path.read_text()

        assert 'PARAMETER stop "<|endoftext|>"' in content
        assert 'PARAMETER stop "</s>"' in content

    def test_generate_modelfile_returns_path(self, tmp_path: Path) -> None:
        """generate_modelfile() が Path オブジェクトを返すことを確認する。"""
        converter = _make_converter(tmp_path)
        output_dir = Path(converter.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = output_dir / "model_f16.gguf"
        gguf_path.touch()

        result = converter.generate_modelfile(gguf_path)

        assert isinstance(result, Path)
