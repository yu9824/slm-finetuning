"""scripts/train.py: 学習パイプラインのエントリポイント。

OfflineGuard → ConfigManager → TrainingCore の順でパイプラインを組み立て、
SFT ファインチューニングを実行する。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# 重量級ライブラリのトップレベルインポートを避けるため、
# enforce_offline / peft / transformers 等は main() 内で遅延インポートするが、
# unittest.mock.patch がモジュール名前空間でパッチできるよう、
# 以下のシンボルはモジュールスコープで公開する。
from src.utils.offline import enforce_offline  # noqa: E402
from src.training.config import load_config, save_config_snapshot  # noqa: E402
from src.training.core import run_training_trial  # noqa: E402
from src.model.loader import load_model_and_tokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI の引数を解析する。

    Returns:
        解析済みの引数 Namespace。
    """
    parser = argparse.ArgumentParser(
        description="SLM ファインチューニング 学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 設定ファイルのパス（省略可）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="モデルのパス（config の model_path を CLI から上書き）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（config の output_dir を CLI から上書き）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学習率",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="エポック数",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="学習後に完全モデルマージを実行する",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha",
    )

    return parser.parse_args()


def main() -> None:
    """学習パイプラインのエントリポイント。

    OfflineGuard → ConfigManager → TrainingCore の順で実行する。
    """
    import torch

    enforce_offline()

    args = parse_args()

    # CLI オーバーライド辞書を構築（None 値は除外）
    cli_overrides: dict[str, Any] = {}
    if args.model_path:
        cli_overrides["model_path"] = args.model_path
    if args.output_dir:
        cli_overrides["output_dir"] = args.output_dir
    if args.learning_rate:
        cli_overrides["learning_rate"] = args.learning_rate
    if args.num_train_epochs:
        cli_overrides["num_train_epochs"] = args.num_train_epochs
    if args.lora_r:
        cli_overrides["lora.r"] = args.lora_r
    if args.lora_alpha:
        # _apply_nested_overrides の実装上 lora.lora_alpha として渡す
        cli_overrides["lora.lora_alpha"] = args.lora_alpha

    config = load_config(args.config, cli_overrides)

    # 出力ディレクトリを作成
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 設定スナップショットを保存
    save_config_snapshot(config, config.output_dir)

    # 学習を実行
    result = run_training_trial(config)
    print(f"Training completed. eval_loss={result.eval_loss:.4f}")

    # --merge フラグが指定された場合: アダプターをベースモデルにマージして保存
    if args.merge:
        merged_dir = str(Path(config.output_dir) / "merged")
        base_model, tokenizer = load_model_and_tokenizer(
            config.model_path,
            use_qlora=False,
            gradient_checkpointing=False,
        )
        peft_model = PeftModel.from_pretrained(base_model, config.output_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    # 学習後のデモ推論
    demo_model, demo_tokenizer = load_model_and_tokenizer(
        config.model_path,
        use_qlora=False,
        gradient_checkpointing=False,
    )
    demo_peft = PeftModel.from_pretrained(demo_model, config.output_dir)
    sample_prompt = "日本語でこんにちはと言ってください。"
    inputs = demo_tokenizer(sample_prompt, return_tensors="pt").to(demo_peft.device)
    with torch.no_grad():
        outputs = demo_peft.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = demo_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    print(f"Demo inference:\n  Prompt: {sample_prompt}\n  Output: {generated}")


if __name__ == "__main__":
    main()
