"""GGUF変換スクリプト。

llama.cpp を使用してマージ済み HuggingFace モデルを GGUF 形式に変換し、
Ollama 用 Modelfile を生成する。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class GGUFConverter:
    """HuggingFace モデルを GGUF 形式に変換するクラス。

    Args:
        llama_cpp_path: llama.cpp リポジトリのディレクトリパス。
        model_path: 変換元のマージ済み HuggingFace モデルのパス。
        output_dir: 変換後の GGUF ファイルを出力するディレクトリ。

    Raises:
        FileNotFoundError: llama_cpp_path または convert_hf_to_gguf.py が存在しない場合。
    """

    def __init__(self, llama_cpp_path: str, model_path: str, output_dir: str) -> None:
        llama_cpp = Path(llama_cpp_path)

        if not llama_cpp.exists():
            raise FileNotFoundError(
                f"llama.cpp not found at {llama_cpp_path}.\n"
                "Please clone it first:\n"
                f"  git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path}\n"
                f"  cd {llama_cpp_path} && make"
            )

        convert_script = llama_cpp / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {convert_script}.\n"
                "Please clone it first:\n"
                f"  git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path}\n"
                f"  cd {llama_cpp_path} && make"
            )

        self.llama_cpp_path = str(llama_cpp)
        self.model_path = model_path
        self.output_dir = output_dir

    def convert(self, quant_type: str = "q4_k_m") -> Path:
        """HuggingFace モデルを GGUF 形式に変換する。

        ステップ 1: convert_hf_to_gguf.py で f16 変換。
        ステップ 2: quant_type が f16 以外の場合、llama-quantize で量子化。

        Args:
            quant_type: 量子化タイプ（例: "q4_k_m", "f16"）。デフォルトは "q4_k_m"。

        Returns:
            変換後の .gguf ファイルのパス。

        Raises:
            RuntimeError: subprocess の実行に失敗した場合。
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        f16_gguf = output_dir / "model_f16.gguf"

        # ステップ1: f16 変換
        try:
            subprocess.run(
                [
                    sys.executable,
                    f"{self.llama_cpp_path}/convert_hf_to_gguf.py",
                    self.model_path,
                    "--outtype",
                    "f16",
                    "--outfile",
                    str(f16_gguf),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"f16 変換に失敗しました: {exc}"
            ) from exc

        if quant_type == "f16":
            return f16_gguf

        # ステップ2: 量子化
        quant_upper = quant_type.upper()
        quantized_gguf = output_dir / f"model_{quant_type}.gguf"

        try:
            subprocess.run(
                [
                    f"{self.llama_cpp_path}/llama-quantize",
                    str(f16_gguf),
                    str(quantized_gguf),
                    quant_upper,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"量子化に失敗しました ({quant_type}): {exc}"
            ) from exc

        return quantized_gguf

    def generate_modelfile(
        self, gguf_path: Path, model_name: str = "sarashina-ft"
    ) -> Path:
        """Ollama 用 Modelfile を生成する。

        Args:
            gguf_path: GGUF ファイルのパス。
            model_name: Ollama 用モデル名。デフォルトは "sarashina-ft"。

        Returns:
            生成した Modelfile のパス。
        """
        modelfile_path = Path(self.output_dir) / "Modelfile"
        content = (
            f"FROM ./{gguf_path.name}\n"
            "\n"
            'PARAMETER stop "<|endoftext|>"\n'
            'PARAMETER stop "</s>"\n'
        )
        modelfile_path.write_text(content, encoding="utf-8")
        return modelfile_path


def main() -> None:
    """CLI エントリーポイント。"""
    from src.utils.offline import enforce_offline

    enforce_offline()

    parser = argparse.ArgumentParser(
        description="HuggingFace モデルを GGUF 形式に変換し、Ollama 用 Modelfile を生成する。"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="マージ済み HuggingFace モデルのパス。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="出力ディレクトリ。",
    )
    parser.add_argument(
        "--quant-type",
        default="q4_k_m",
        help="量子化タイプ（例: q4_k_m, f16）。デフォルト: q4_k_m。",
    )
    parser.add_argument(
        "--llama-cpp-path",
        default="./llama.cpp",
        help="llama.cpp ディレクトリのパス。デフォルト: ./llama.cpp。",
    )
    parser.add_argument(
        "--model-name",
        default="sarashina-ft",
        help="Ollama 用モデル名。デフォルト: sarashina-ft。",
    )
    args = parser.parse_args()

    converter = GGUFConverter(
        llama_cpp_path=args.llama_cpp_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    print(f"変換開始: {args.model_path} -> {args.output_dir} (quant_type={args.quant_type})")
    gguf_path = converter.convert(quant_type=args.quant_type)
    print(f"GGUF 変換完了: {gguf_path}")

    modelfile_path = converter.generate_modelfile(gguf_path, model_name=args.model_name)
    print(f"Modelfile 生成完了: {modelfile_path}")

    print("\n次のステップ:")
    print(f"  cd {args.output_dir}")
    print(f"  ollama create {args.model_name} -f Modelfile")
    print(f"  ollama run {args.model_name}")


if __name__ == "__main__":
    main()
