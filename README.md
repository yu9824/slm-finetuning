# SLM Fine-tuning

日本語 SLM「sarashina-2.2:3b」を対象とした LoRA ベースのファインチューニングテンプレート。  
**完全オフライン環境**での動作を前提に設計しており、データ準備・学習・HPO・可視化・GGUF 変換まで一通り体験できる実践的なサンプルです。

## 特徴

- **LoRA / QLoRA** による省メモリファインチューニング（24GB 以下の GPU 対応）
- **完全オフライン動作**（`TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` を強制設定）
- **YAML 設定ファイル**によるハイパーパラメータ管理（CLI 引数でオーバーライド可）
- **Optuna / Ray Tune** によるハイパーパラメータ最適化（HPO）
- **TensorBoard / wandb（オフライン）** による学習モニタリング
- **GGUF 変換 + Ollama デプロイ**フロー

## ディレクトリ構成

```
.
├── configs/
│   ├── train_config.yaml     # 学習設定テンプレート
│   └── hpo_config.yaml       # HPO 設定テンプレート
├── data/
│   ├── README.md             # データ形式の説明
│   ├── dummy_train.jsonl     # 学習用ダミーデータ（25件）
│   └── dummy_eval.jsonl      # 評価用ダミーデータ（7件）
├── scripts/
│   ├── train.py              # 学習エントリポイント
│   ├── hpo.py                # ハイパーパラメータ最適化
│   ├── inference.py          # 推論サンプル
│   └── convert_to_gguf.py   # GGUF 変換 + Modelfile 生成
├── src/
│   ├── data/processor.py     # データ前処理
│   ├── model/loader.py       # モデル・トークナイザーのロード
│   ├── training/
│   │   ├── config.py         # 設定管理
│   │   └── core.py           # 学習ループ
│   └── utils/offline.py      # オフライン強制設定
├── requirements.txt
└── requirements-ray.txt      # Ray Tune 使用時の追加依存
```

## セットアップ

### 1. オンライン環境での事前準備

```bash
# モデル重みのダウンロード
huggingface-cli download sbintuitions/sarashina2.2-3b --local-dir /path/to/sarashina-2.2-3b

# Python パッケージのダウンロード（オフライン転送用）
pip download -r requirements.txt -d ./wheels

# llama.cpp のビルド（GGUF 変換に必要）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
```

### 2. オフライン環境でのインストール

```bash
# パッケージのオフラインインストール
pip install --no-index --find-links ./wheels -r requirements.txt

# Ray Tune を使う場合
pip install --no-index --find-links ./wheels -r requirements-ray.txt
```

### オフライン環境チェックリスト

- [ ] モデル重みファイル（`/path/to/sarashina-2.2-3b/`）
- [ ] Python パッケージ（`./wheels/`）
- [ ] llama.cpp リポジトリ（GGUF 変換用）
- [ ] Ollama バイナリ（推論サーバー用）

## 使い方

### 学習

```bash
# 設定ファイルを使って学習
python scripts/train.py --config configs/train_config.yaml

# CLI 引数でオーバーライド
python scripts/train.py --config configs/train_config.yaml \
  --model-path /path/to/sarashina-2.2-3b \
  --output-dir outputs/my-experiment \
  --learning-rate 5e-5

# 学習後にアダプターをベースモデルとマージ
python scripts/train.py --config configs/train_config.yaml --merge
```

`configs/train_config.yaml` の主要パラメータ：

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `model_path` | （必須） | ローカルモデルパス |
| `output_dir` | `outputs/sarashina-ft` | 出力先 |
| `learning_rate` | `1e-4` | 学習率 |
| `num_train_epochs` | `3` | エポック数 |
| `lora.r` | `16` | LoRA rank |
| `lora.use_qlora` | `false` | QLoRA（4-bit 量子化） |

### 推論

```bash
python scripts/inference.py \
  --model-path /path/to/sarashina-2.2-3b \
  --adapter-path outputs/sarashina-ft \
  --prompt "日本語でこんにちはと言ってください。" \
  --max-new-tokens 100 \
  --temperature 0.7
```

### ハイパーパラメータ最適化（HPO）

```bash
# Optuna バックエンド（デフォルト）
python scripts/hpo.py --config configs/hpo_config.yaml --n-trials 20

# Ray Tune バックエンド
python scripts/hpo.py --config configs/hpo_config.yaml --backend ray --n-trials 20
```

探索空間（`configs/hpo_config.yaml` で変更可）：

| パラメータ | 探索範囲 |
|---|---|
| `learning_rate` | `[1e-5, 1e-3]`（対数一様分布） |
| `lora_r` | `[4, 8, 16, 32]`（カテゴリカル） |
| `lora_alpha` | `[16, 32, 64]`（カテゴリカル） |
| `per_device_train_batch_size` | `[1, 2, 4]`（カテゴリカル） |

最良パラメータは `hpo_results/best_params.yaml` に保存されます。

### 学習の可視化

```bash
tensorboard --logdir runs/
```

ブラウザで `http://localhost:6006` を開いて確認します。  
ログには train loss / eval loss / 学習率 / 勾配ノルムが記録されます。

wandb を使う場合はオフラインモードで動作します：

```bash
WANDB_MODE=offline python scripts/train.py --config configs/train_config.yaml
```

### GGUF 変換と Ollama デプロイ

```bash
# マージ済みモデルを GGUF 変換（q4_k_m 量子化）
python scripts/convert_to_gguf.py \
  --model-path outputs/sarashina-ft/merged \
  --output-dir outputs/gguf \
  --llama-cpp-path /path/to/llama.cpp \
  --quantization q4_k_m

# Ollama へ登録
ollama create sarashina-ft -f outputs/gguf/Modelfile

# 推論
ollama run sarashina-ft
```

## データ形式

JSONL 形式（1行1サンプル）。詳細は [data/README.md](data/README.md) を参照してください。

```json
{"instruction": "日本の首都はどこですか？", "output": "日本の首都は東京です。"}
{"instruction": "次の文章を要約してください。", "input": "（長文）", "output": "（要約）"}
```

| フィールド | 必須 | 説明 |
|---|---|---|
| `instruction` | 必須 | モデルへの指示・質問 |
| `input` | 任意 | 追加コンテキスト |
| `output` | 必須 | 期待される出力 |

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `torch` | ベースフレームワーク |
| `transformers` | モデル・トークナイザー |
| `trl` | SFTTrainer |
| `peft` | LoRA / QLoRA |
| `bitsandbytes` | 4-bit 量子化 |
| `optuna` | HPO |
| `tensorboard` | 学習可視化 |
| `datasets` | データ読み込み |
| `accelerate` | マルチ GPU サポート |
| `wandb` | 実験トラッキング（オフライン） |
