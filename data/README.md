# データセット

このディレクトリには、SLM（Small Language Model）のファインチューニングに使用するデータセットが格納されています。

## データ形式

### JSONL（JSON Lines）形式

各ファイルは **JSONL 形式**（1行1サンプル）を採用しています。各行は独立した JSON オブジェクトであり、改行で区切られます。

### フィールド説明

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `instruction` | string | 必須 | モデルへの指示・質問文 |
| `input` | string | 任意 | 指示に付随するコンテキスト・入力テキスト（省略可） |
| `output` | string | 必須 | 期待される出力・回答文 |

`input` フィールドは省略可能です。指示だけで完結するサンプルでは省略しています。

### サンプル例

```jsonl
{"instruction": "日本の首都はどこですか？", "output": "日本の首都は東京です。東京は1869年に明治天皇が移り住んで以来、日本の政治・経済・文化の中心地となっています。"}
{"instruction": "次の文章を100字程度に要約してください。", "input": "人工知能（AI）は...", "output": "人工知能はコンピュータが人間のように学習・推論する技術で..."}
```

## ファイル一覧

| ファイル名 | 件数 | 用途 | カテゴリ |
|---|---|---|---|
| `dummy_train.jsonl` | 25件 | 学習用 | 一般知識・文章生成・要約・質問応答・翻訳（各5件） |
| `dummy_eval.jsonl` | 7件 | 評価用 | 複数カテゴリ混在 |

### カテゴリ構成（`dummy_train.jsonl`）

- **一般知識・雑学**（5件）：首都・光合成・富士山・アンペア・江戸時代など
- **文章生成・作文**（5件）：詩・お祝いメッセージ・意見文・物語・描写文など
- **要約**（5件）：長文・会話文・ニュース記事などの要約タスク
- **質問応答**（5件）：事実質問・説明・手順案内など
- **翻訳・言語変換**（5件）：日英・英日翻訳、敬語変換、文体変換など

## 使用方法

### Python でのデータ読み込み

```python
import json

def load_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_data = load_jsonl("data/dummy_train.jsonl")
eval_data = load_jsonl("data/dummy_eval.jsonl")

print(f"学習データ: {len(train_data)} 件")
print(f"評価データ: {len(eval_data)} 件")
```

### プロンプトへの変換例

```python
def format_prompt(sample: dict) -> str:
    if sample.get("input"):
        return f"### 指示:\n{sample['instruction']}\n\n### 入力:\n{sample['input']}\n\n### 応答:\n"
    else:
        return f"### 指示:\n{sample['instruction']}\n\n### 応答:\n"
```
