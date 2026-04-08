"""scripts/inference.py: 保存済みLoRAアダプターをロードして推論を実行するCLIスクリプト。

Usage:
    python scripts/inference.py \
        --base-model-path /path/to/base_model \
        --adapter-path /path/to/adapter \
        --prompt "質問テキスト"
"""

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.offline import enforce_offline  # noqa: E402

# Heavy imports are placed after path setup but before enforce_offline call in
# run_inference; top-level import of the guard function is fine since it has no
# side-effects at import time.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def run_inference(
    base_model_path: str,
    adapter_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """アダプターをロードし、プロンプトに対して生成テキストを返す。

    Args:
        base_model_path: ベースモデルのディレクトリパス。
        adapter_path: LoRAアダプターのディレクトリパス。
        prompt: ユーザー入力プロンプト。
        max_new_tokens: 生成する最大トークン数。
        temperature: サンプリング温度。
        do_sample: サンプリングを行うかどうか。

    Returns:
        モデルが生成したテキスト（入力プロンプト部分を除く）。

    Raises:
        FileNotFoundError: base_model_path または adapter_path が存在しない場合。
    """
    base_model_dir = Path(base_model_path)
    if not base_model_dir.exists():
        raise FileNotFoundError(
            f"base_model_path が見つかりません: {base_model_path}"
        )

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter_path が見つかりません: {adapter_path}"
        )

    # ベースモデルとトークナイザーのロード
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # LoRAアダプターのロード
    model = PeftModel.from_pretrained(model, adapter_path)

    # チャットテンプレートの適用
    messages = [
        {"role": "system", "content": "あなたは誠実で優秀なアシスタントです。"},
        {"role": "user", "content": prompt},
    ]

    try:
        formatted_prompt: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError):
        formatted_prompt = f"[INST] {prompt} [/INST]"

    # トークナイズ
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    input_length: int = input_ids.shape[-1]

    # テキスト生成
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    # 入力部分を除いた生成トークンのみデコード
    generated_text: str = tokenizer.decode(
        output_ids[0][input_length:],
        skip_special_tokens=True,
    )

    return generated_text


def main() -> None:
    """CLIエントリーポイント。argparse で引数を解析して run_inference を呼び出す。"""
    import argparse

    enforce_offline()

    parser = argparse.ArgumentParser(
        description="保存済みLoRAアダプターをロードして推論を実行する。"
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="ベースモデルのディレクトリパス。",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="LoRAアダプターのディレクトリパス。",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="入力プロンプト。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="生成する最大トークン数 (default: 256)。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="サンプリング温度 (default: 0.7)。",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        default=False,
        help="do_sample=False にする（貪欲デコード）。",
    )

    args = parser.parse_args()

    generated_text = run_inference(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
    )

    print(generated_text)


if __name__ == "__main__":
    main()
