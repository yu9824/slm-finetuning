# リサーチ & 設計判断ログ

---
**目的**: 技術設計に影響する調査結果、アーキテクチャ検討、判断根拠を記録する。

---

## サマリー
- **機能**: `sarashina-finetune`
- **調査スコープ**: 新規機能（グリーンフィールド）- 完全オフライン対応 LLM ファインチューニングテンプレート
- **主要所見**:
  - TRL v1.0 の `SFTTrainer` が LoRA 学習の標準アプローチとして確立されており、PEFT との統合が標準化されている
  - Sarashina-2.2 は LLaMA 系アーキテクチャを採用しており、標準的な LoRA target_modules（attention + MLP 層）が適用可能
  - 完全オフライン運用は `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` 環境変数と `trust_remote_code=True` で実現可能
  - HPO はシングルノード・オフライン要件から Optuna + SQLite が最適（Ray Tune は分散環境向け）
  - GGUF 変換は llama.cpp の `convert_hf_to_gguf.py` + `llama-quantize` の2ステップで実施

## リサーチログ

### Sarashina-2.2 モデルアーキテクチャ

- **調査背景**: オフライン環境での LoRA 設定とチャットテンプレート形式の確認
- **参照情報**:
  - HuggingFace: `sbintuitions/sarashina2.2-3b-instruct-v0.1`
  - SB Intuitions 公式リポジトリのモデルカード
- **所見**:
  - LLaMA 系アーキテクチャを採用（transformer ブロック構造が類似）
  - チャットテンプレートはロールベース形式: `{"role": "user"/"assistant", "content": "..."}`
  - `tokenizer_config.json` に `apply_chat_template()` 対応テンプレートを内包
  - LoRA target_modules の推奨: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
    - attention 層のみの最小構成: `["q_proj", "v_proj"]`
    - 精度優先の拡張構成: attention + MLP 全層
  - 既存ファインチューニング済みアダプターが Hub 上に存在
- **アーキテクチャへの影響**:
  - チャットテンプレートは `tokenizer.apply_chat_template()` を直接使用できる
  - instruction/input/output → messages 形式への変換ロジックが必要
  - `trust_remote_code=True` が必須（カスタムコードを含む可能性）

---

### TRL SFTTrainer（v1.0）

- **調査背景**: 学習ループの実装方針確定
- **参照情報**: HuggingFace TRL 公式ドキュメント、TRL v1.0 リリースノート（2026年4月）
- **所見**:
  - `trl.SFTTrainer` + `trl.SFTConfig` が標準 API として統一
  - `transformers.Trainer` のラッパーであり、全属性・メソッドを継承
  - PEFT との統合: `peft_config` パラメータで LoraConfig を直接渡せる
  - 対応データセット形式:
    - 会話形式: `{"messages": [{"role": ..., "content": ...}]}`
    - 補完形式: `{"prompt": ..., "completion": ...}`
    - 言語モデル形式: `{"text": ...}`
  - `assistant_only_loss=True` でアシスタント応答部分のみの損失計算が可能
  - `packing=True` でシーケンスパッキングによる学習効率向上
  - Unsloth / Liger Kernel オプション統合（速度・メモリ改善）
- **アーキテクチャへの影響**:
  - `transformers.Trainer` ではなく `trl.SFTTrainer` を採用
  - TensorBoard 統合は `SFTConfig(report_to=["tensorboard"])` で設定
  - gradient_checkpointing はデフォルト有効

---

### PEFT LoRA / QLoRA ベストプラクティス

- **調査背景**: GPU メモリ 24GB 以下でのファインチューニング設定の最適化
- **参照情報**: HuggingFace PEFT 公式ドキュメント、BitsAndBytes 量子化ガイド
- **所見**:
  - **LoRA ハイパーパラメータ推奨値**（LLaMA 系 3B モデル）:
    - `r`: 16（標準）、複雑タスクは 32〜64
    - `lora_alpha`: r の 2 倍（例: r=16 → alpha=32）
    - `lora_dropout`: 0.05〜0.1
    - `bias`: "none"
    - `task_type`: "CAUSAL_LM"
  - **学習率**: 1e-4〜5e-4（フルファインチューニングの約 10 倍）
  - **QLoRA 設定**（BitsAndBytesConfig）:
    - `load_in_4bit=True`
    - `bnb_4bit_quant_type="nf4"`（Normal Float 4、学習向け）
    - `bnb_4bit_compute_dtype=torch.bfloat16`
    - `bnb_4bit_use_double_quant=True`（二重量子化、0.4 ビット/パラメータ節約）
  - QLoRA によりメモリを 10〜20 倍削減しながら品質の 90〜95% を保持
  - マージ: `model.merge_and_unload()` でアダプターをベースモデルに統合
- **アーキテクチャへの影響**:
  - QLoRA はオプション設定（設定ファイルで `use_qlora: bool` として制御）
  - QLoRA 使用時は `device_map="auto"` が必須
  - gradient_checkpointing と QLoRA の併用は `use_gradient_checkpointing="unsloth"` の考慮が必要

---

### HPO バックエンド比較（Optuna vs Ray Tune）

- **調査背景**: 完全オフライン・シングルノード環境での HPO 実装選定
- **参照情報**: Optuna 公式ドキュメント、Ray Tune 公式ドキュメント
- **所見**:

  | 観点 | Optuna | Ray Tune |
  |------|--------|----------|
  | シングルノード適合性 | ◎ SQLite ストレージで完結 | ○ ローカルモード対応 |
  | オフライン対応 | ◎ 外部接続不要 | △ Ray Dashboard は外部接続可能性あり |
  | 並列探索 | △ SQLite では推奨しない（ロック競合） | ◎ マルチプロセス自動並列化 |
  | セットアップ複雑さ | 低 | 中〜高 |
  | 試行永続化 | ◎ SQLite ファイルに自動保存 | ○ Ray ローカルディレクトリ |
  | Transformers 統合 | ◎ `trainer.hyperparameter_search()` 対応 | ◎ 同様に対応 |

  - Optuna の SQLite storage 注意点: 並列実行時はタイムアウト設定が必要 (`timeout=20.0`)
  - Ray Tune は `ray.init(local_mode=True)` でローカル単体動作確認済み
  - Ray Dashboard は `ray.init(dashboard_host="127.0.0.1")` でローカル限定にできる
- **アーキテクチャへの影響**:
  - デフォルトバックエンドを Optuna とし、設定ファイルで Ray Tune に切り替え可能な設計
  - Optuna + SQLite ストレージでオフライン動作を保証
  - `transformers.Trainer.hyperparameter_search()` を通じた統合も検討するが、柔軟性から独立スクリプト採用

---

### llama.cpp GGUF 変換

- **調査背景**: ファインチューニング済みモデルの Ollama デプロイ手順の確立
- **参照情報**: llama.cpp GitHub リポジトリ、convert_hf_to_gguf.py ドキュメント
- **所見**:
  - **2ステップ変換プロセス**:
    1. HF → GGUF（FP16）: `python convert_hf_to_gguf.py [MODEL_PATH] --outtype f16 --outfile model.gguf`
    2. 量子化: `./llama-quantize model.gguf model-q4km.gguf Q4_K_M`
  - `convert_hf_to_gguf.py` の主要引数:
    - `dir_model`: 入力 HuggingFace モデルディレクトリ
    - `--outfile`: 出力 GGUF ファイルパス
    - `--outtype`: f32/f16/bf16/q8_0（変換時の精度）
    - `--model-name`: モデル識別名
  - 量子化タイプ推奨:
    - `Q4_K_M`: バランス型（品質・サイズ）
    - `Q5_K_M`: 高品質
    - `Q8_0`: ほぼ無損失
    - `F16`: 完全精度（サイズ大）
  - Ollama Modelfile 形式: `FROM ./model.gguf\nPARAMETER stop "..."`
- **アーキテクチャへの影響**:
  - llama.cpp のパスは設定ファイルまたは引数で指定
  - 変換スクリプトは Python subprocess で `convert_hf_to_gguf.py` を呼び出す
  - llama.cpp が見つからない場合のエラーメッセージに取得手順を含める

---

### TensorBoard オフライン動作

- **調査背景**: オフライン環境での学習可視化の実現方法確認
- **参照情報**: Transformers TrainingArguments ドキュメント
- **所見**:
  - `TrainingArguments(report_to=["tensorboard"], logging_dir="./runs/")` で完全ローカル動作
  - `TensorBoardCallback` はパッケージが存在すれば自動登録
  - ログはファイルシステムに書き出され、インターネット接続不要
  - 起動コマンド: `tensorboard --logdir ./runs`（ローカルブラウザで確認）
  - Weights & Biases オフラインモード: `WANDB_MODE=offline` でローカルログ保存
- **アーキテクチャへの影響**:
  - TensorBoard をデフォルトとし、W&B はオプション（`WANDB_MODE=offline` 設定を明記）
  - `grad_norm` の記録は `report_to="tensorboard"` + `logging_steps` で自動的に取得可能

---

## アーキテクチャパターン評価

| オプション | 説明 | 強み | リスク/制約 | 備考 |
|-----------|------|------|------------|------|
| モノリシックスクリプト | すべてを単一 train.py に実装 | シンプル | 再利用性・テスト性が低い | 学習テンプレートとしては不適切 |
| 機能別モジュール分割 | src/ にコアロジック、scripts/ にエントリポイント | 再利用性高・テスト可能 | やや複雑 | **採用** |
| パッケージ化（pip install） | setup.py / pyproject.toml でパッケージ化 | 最高の再利用性 | オーバーエンジニアリング | 学習テンプレートには不要 |

**採用パターン**: 機能別モジュール分割（`src/` + `scripts/` 構成）

- `src/`: 再利用可能なコアロジック（モデルロード、データ処理、設定管理）
- `scripts/`: エントリポイント（train.py、hpo.py、inference.py、convert_to_gguf.py）
- `configs/`: YAML 設定ファイル
- `data/`: ダミーデータセット

---

## 設計判断

### 判断: チャットテンプレートの適用方式

- **背景**: instruction/input/output → sarashina-2.2 のチャット形式への変換
- **検討した選択肢**:
  1. 手動フォーマット文字列（Alpaca 形式テンプレート）
  2. `tokenizer.apply_chat_template()` を使用
- **採用アプローチ**: `tokenizer.apply_chat_template()` を使用し、messages リストを構築してから適用
- **根拠**: モデル固有のテンプレートを自動適用でき、将来の他モデルへの移植性が高い
- **トレードオフ**: テンプレートのカスタマイズが若干複雑になるが、正確性が向上
- **実装確認事項**: `input` フィールドが空の場合の messages 構築ロジック

---

### 判断: HPO バックエンドのデフォルト選択

- **背景**: 完全オフライン・シングルノード環境での HPO
- **検討した選択肢**:
  1. Optuna のみ
  2. Ray Tune のみ
  3. 設定ファイルで切り替え可能（両対応）
- **採用アプローチ**: 設定ファイルで切り替え可能、デフォルトは Optuna
- **根拠**: 要件 8.1 が「いずれかをバックエンドとして選択できる」と明示；Optuna はオフライン要件に最適
- **トレードオフ**: 実装コスト増加、ただし学習テンプレートとしての教育的価値が高い
- **実装確認事項**: Ray Tune の `local_mode=True` でのオフライン動作検証

---

### 判断: GGUF 変換の実装方式

- **背景**: llama.cpp の Python スクリプトの呼び出し方法
- **検討した選択肢**:
  1. subprocess で `convert_hf_to_gguf.py` を直接呼び出し
  2. llama.cpp を Python ライブラリとしてインポート
- **採用アプローチ**: subprocess による外部スクリプト呼び出し
- **根拠**: llama.cpp に Python パッケージ API が存在しない；オフライン環境でビルド済みバイナリを想定
- **トレードオフ**: llama.cpp のパス依存性があるが、エラーハンドリングで取得手順を案内することで対処
- **実装確認事項**: `convert_hf_to_gguf.py` の引数仕様（バージョンによる変動の可能性）

---

## リスクと軽減策

- **QLoRA + gradient_checkpointing の互換性問題** — `use_gradient_checkpointing="unsloth"` または単純な無効化で対処；設定ファイルで制御可能にする
- **llama.cpp バージョン差異による引数変動** — `convert_hf_to_gguf.py --help` の出力確認手順をドキュメントに明記
- **Sarashina-2.2 のチャットテンプレート詳細** — オフライン環境では `tokenizer_config.json` を直接確認；ダミーデータで検証
- **Ray Tune ローカルモードの外部通信** — `RAY_DISABLE_IMPORT_WARNING=1` と `dashboard_host` 設定でローカル限定化
- **SQLite 並列 HPO のロック競合** — Optuna の `n_jobs=1`（シングルスレッド）をデフォルトとし、並列化はドキュメントで注意喚起

## 参考リンク

- [Sarashina-2.2 3B Instruct on HuggingFace](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1) — モデルカードとアーキテクチャ詳細
- [TRL SFTTrainer ドキュメント](https://huggingface.co/docs/trl/sft_trainer) — v1.0 API リファレンス
- [PEFT LoRA ドキュメント](https://huggingface.co/docs/peft/developer_guides/lora) — LoraConfig パラメータ
- [BitsAndBytes 量子化](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) — QLoRA 設定
- [Optuna RDB Storage](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) — SQLite バックエンド
- [Ray Tune ドキュメント](https://docs.ray.io/en/latest/tune/index.html) — ローカルモード設定
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) — convert_hf_to_gguf.py 使用方法
- [Transformers TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) — TensorBoard 統合
