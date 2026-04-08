"""Offline guard utilities.

Import this module and call ``enforce_offline()`` as the very first statement
of every script's ``main()`` — before any HuggingFace or W&B imports — to
prevent accidental outbound network calls.
"""

import logging
import os

logger = logging.getLogger(__name__)

_OFFLINE_VARS: dict[str, str] = {
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "offline",
}


def enforce_offline() -> None:
    """スクリプト起動時に呼び出し、HuggingFace Hub / W&B への外部通信を遮断する。

    環境変数を上書き設定するため、import 直後・他の import より前に呼び出すこと。

    Post-conditions:
        ``TRANSFORMERS_OFFLINE``, ``HF_DATASETS_OFFLINE``, ``WANDB_MODE`` が
        ``os.environ`` に設定済みであること。

    Note:
        既存の環境変数を問答無用で上書きする（ユーザー設定より強制設定を優先）。
    """
    os.environ.update(_OFFLINE_VARS)

    logger.info(
        "[OfflineGuard] オフラインモード有効: "
        "TRANSFORMERS_OFFLINE=%s, HF_DATASETS_OFFLINE=%s, WANDB_MODE=%s",
        os.environ["TRANSFORMERS_OFFLINE"],
        os.environ["HF_DATASETS_OFFLINE"],
        os.environ["WANDB_MODE"],
    )
