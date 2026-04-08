"""Core training loop for SFT fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import trl
from trl import SFTConfig, SFTTrainer

from src.data.processor import DatasetProcessor
from src.model.loader import apply_lora, load_model_and_tokenizer
from src.training.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """学習トライアルの結果。

    Attributes:
        eval_loss: 評価データセットに対する損失。eval_dataset が None の場合は NaN。
        model_path: 学習済みアダプターの保存先ディレクトリ。
    """

    eval_loss: float
    model_path: str


def run_training_trial(config: TrainConfig) -> TrialResult:
    """TrainConfig に基づきモデルロード・データ準備・SFTTrainer 実行を行い、TrialResult を返す。

    train.py からはフルトレーニングとして、hpo.py からは各トライアルとして呼び出される。

    Args:
        config: 学習設定。

    Returns:
        eval_loss と model_path を含む TrialResult。
        eval_dataset が None の場合、eval_loss は float("nan")。

    Raises:
        FileNotFoundError: config.model_path が存在しない場合。
        RuntimeError: 学習中にエラーが発生した場合。
    """
    # 1. モデルとトークナイザーのロード
    model, tokenizer = load_model_and_tokenizer(
        config.model_path,
        use_qlora=config.lora.use_qlora,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    logger.info("Model and tokenizer loaded from: %s", config.model_path)

    # 2. LoRA の適用
    model = apply_lora(model, config.lora)
    logger.info("LoRA adapter applied.")

    # 3. データセットの取得
    train_dataset, eval_dataset = DatasetProcessor(config.data).get_train_eval_datasets()
    logger.info(
        "Datasets loaded. train=%d samples, eval=%s samples",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else "N/A",
    )

    # 4. SFTConfig の構築
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        max_seq_length=config.data.max_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        report_to=config.report_to,
        logging_dir=config.logging_dir,
    )

    # 5. SFTTrainer の作成
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 6. 学習の実行
    try:
        trainer.train()
        logger.info("Training completed.")
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc

    # 7. eval_loss の取得
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss: float = eval_metrics["eval_loss"]
        logger.info("Evaluation loss: %f", eval_loss)
    else:
        eval_loss = float("nan")
        logger.info("No eval dataset provided; eval_loss set to NaN.")

    # 8. アダプターの保存
    trainer.model.save_pretrained(config.output_dir)
    logger.info("Adapter saved to: %s", config.output_dir)

    # 9. TrialResult を返す
    return TrialResult(eval_loss=eval_loss, model_path=config.output_dir)
