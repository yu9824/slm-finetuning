"""Model and tokenizer loading utilities for LoRA / QLoRA fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from peft import PeftModel, TaskType, get_peft_model
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from src.training.config import LoRAConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    use_qlora: bool = False,
    gradient_checkpointing: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """指定パスからローカルモデルとトークナイザーをロードする。

    Args:
        model_path: ローカルモデルディレクトリのパス。
        use_qlora: QLoRA (4-bit量子化) を使用するか。
        gradient_checkpointing: gradient checkpointing を有効化するか。

    Returns:
        (model, tokenizer) のタプル。

    Raises:
        FileNotFoundError: model_path が存在しない場合。
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    quantization_config: BitsAndBytesConfig | None = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA enabled: loading model in 4-bit quantization.")

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if use_qlora and gradient_checkpointing:
        model.enable_input_require_grads()
        logger.info("Enabled input require grads for QLoRA + gradient checkpointing compatibility.")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token was None; set to eos_token (%s).", tokenizer.eos_token)

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: "LoRAConfig",
) -> PeftModel:
    """LoRAアダプターをモデルに適用し、ベースモデルを凍結する。

    Args:
        model: ベースモデル。
        lora_config: LoRA設定データクラス。

    Returns:
        LoRAアダプターが適用された PeftModel。
    """
    peft_lora_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
    )

    peft_model: PeftModel = get_peft_model(model, peft_lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
