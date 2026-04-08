# 実装計画

## タスク一覧

---

- [x] 1. オフラインガードと設定管理の実装
- [x] 1.1 OfflineGuard の実装
  - `src/utils/offline.py` に `enforce_offline()` 関数を実装する
  - `TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`WANDB_MODE=offline` を `os.environ` に強制設定する
  - ネットワーク接続が必要な処理が発生した場合に明示的エラーを出力して中断する仕組みを含める
  - _Requirements: 1.4, 1.5, 9.6, 9.7, 11.5_

- [x] 1.2 設定データクラスと ConfigManager の実装
  - `src/training/config.py` に `TrainConfig`・`LoRAConfig`・`DataConfig`・`HPOConfig` データクラスを定義する
  - YAML / JSON 設定ファイルのロードと CLI 引数（argparse）による上書き（CLI 優先）を実装する
  - `gradient_checkpointing` および `report_to="tensorboard"` を設定項目として含める
  - 実験開始時に設定スナップショットを出力ディレクトリに保存する `save_snapshot()` を実装する
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 9.4_

- [x] 1.3 YAML 設定テンプレートファイルの作成
  - `configs/train_config.yaml` に `learning_rate`・`num_train_epochs`・`per_device_train_batch_size`・`gradient_accumulation_steps` などの標準ハイパーパラメータを定義する
  - LoRA 設定（`r`・`lora_alpha`・`target_modules`・`use_qlora`）を設定ファイルに含める
  - _Requirements: 3.1, 3.2, 4.1_

---

- [x] 2. モデルローダーの実装
- [x] 2.1 ローカルモデルとトークナイザーのロード
  - `src/model/loader.py` に `load_model_and_tokenizer()` を実装する
  - 設定ファイルまたはCLI引数で指定されたローカルパスから `AutoModelForCausalLM` と `AutoTokenizer` をロードする
  - `trust_remote_code=True` を明示的に設定してロードする
  - 指定パスにモデルファイルが存在しない場合はエラーメッセージを出力して処理を終了する
  - GPU が利用可能な場合は自動的に GPU を使用し、CPU フォールバックも許容する
  - _Requirements: 1.1, 1.2, 1.3, 3.5_

- [x] 2.2 LoRA / QLoRA アダプターの適用
  - `src/model/loader.py` に `apply_lora()` を実装し、`LoraConfig` を使って LoRA アダプターをモデルに適用する
  - `r`・`lora_alpha`・`target_modules` を `LoRAConfig` から受け取るようにする
  - ベースモデルのパラメータを凍結し、アダプター部分のみを学習対象とする
  - `use_qlora=True` の場合は `BitsAndBytesConfig` による 4-bit 量子化を有効化する
  - _Requirements: 3.1, 3.2, 3.3_

---

- [x] 3. データセットプロセッサーの実装
- [x] 3.1 JSONL データセットの読み込みと前処理
  - `src/data/processor.py` に `DatasetProcessor` を実装し、JSONL ファイルを読み込む `load_dataset()` を提供する
  - sarashina-2.2 のチャットテンプレートに従って `instruction`・`input`・`output` フィールドをフォーマットする `format_sample()` を実装する
  - `max_length` を超えるシーケンスをトランケートする処理を含める
  - 必須フィールド（`instruction`・`output`）が欠落しているサンプルをスキップし、ログに警告を出力する
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3.2 検証セットの損失計算サポート
  - `eval_file` が設定されている場合に検証用データセットをロードして `SFTTrainer` に渡す処理を実装する
  - 学習終了後に検証セットで損失を計算して出力する
  - _Requirements: 2.5_

---

- [x] 4. 学習コアと train.py の実装
- [x] 4.1 TrainingCore（学習ループ）の実装
  - `src/training/core.py` に `TrainingCore` を実装し、`trl.SFTTrainer` を使った学習ループを提供する
  - HPO トライアルから呼び出せるよう `run_training_trial(config, model, datasets)` インターフェースを実装する
  - TensorBoard へのログ出力（`report_to=["tensorboard"]`）を設定し、train loss・eval loss・学習率・grad_norm を各ステップで記録する
  - _Requirements: 3.4, 8.3, 9.1, 9.2, 9.3, 9.4_

- [x] 4.2 train.py（学習パイプラインのエントリポイント）の実装
  - `scripts/train.py` を実装し、`OfflineGuard → ConfigManager → ModelLoader → DatasetProcessor → TrainingCore` の順でパイプラインを組み立てる
  - 学習後に LoRA アダプターを `PeftModel.save_pretrained()` で出力ディレクトリに保存する
  - `merge_and_unload()` による完全モデルマージ保存をオプションとして実装する
  - 出力ディレクトリが存在しない場合に自動的に作成する
  - 学習後にサンプルプロンプトで推論デモを実行して生成テキストを出力する
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.1, 10.1_

---

- [x] 5. (P) ダミーデータセットの作成
  - `data/dummy_train.jsonl` に日本語サンプルを 20 件以上（一般知識・文章生成・要約など複数カテゴリ）含む学習データを作成する
  - `data/dummy_eval.jsonl` に検証用データを 5 件以上作成する
  - `instruction`・`input`（省略可）・`output` フィールドを持つ JSONL 形式とする
  - `data/README.md` にデータ形式の説明を記載する
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

---

- [x] 6. (P) HPORunner の実装
- [x] 6.1 Optuna バックエンドの実装
  - `scripts/hpo.py` に `HPORunner` を実装し、Optuna を使ったハイパーパラメータ探索を提供する
  - `learning_rate`・`lora_r`・`lora_alpha`・`per_device_train_batch_size` を探索空間として定義する
  - Optuna のストレージとして SQLite（ローカルファイル）を使用し、外部データベース接続なしで動作する
  - 評価指標として検証セットの loss を使用し、各トライアルの結果をストレージに保存する
  - `n_trials` を引数で指定できるようにする
  - 探索完了後に最良ハイパーパラメータをファイルに出力する
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6, 8.7_

- [x] 6.2 Ray Tune バックエンドの実装
  - `scripts/hpo.py` の設定切り替えで Ray Tune バックエンドを選択できるようにする
  - `ray.init(local_mode=True)` でローカル単体動作を保証し、外部 Dashboard 接続なしで動作する
  - `HyperOptSearch` または `OptunaSearch` スケジューラと組み合わせた並列探索を実装する
  - _Requirements: 8.1, 8.5, 8.8_

---

- [x] 7. (P) InferenceRunner（推論スクリプト）の実装
  - `scripts/inference.py` を実装し、保存済みアダプターパスを指定してベースモデルにアダプターをロードして推論を実行する
  - `max_new_tokens`・`temperature`・`do_sample` などの生成パラメータを CLI 引数で指定できるようにする
  - _Requirements: 10.2, 10.3_

---

- [x] 8. (P) GGUF 変換スクリプトの実装
  - `scripts/convert_to_gguf.py` に `GGUFConverter` を実装し、`llama.cpp` の `convert_hf_to_gguf.py` を subprocess 経由で呼び出してマージ済み HF モデルを GGUF 形式に変換する
  - 量子化タイプ（`q4_k_m`・`q8_0`・`f16` など）を引数で指定できるようにする
  - 変換後の `.gguf` ファイルを指定の出力ディレクトリに保存する
  - Ollama でホストするための `Modelfile` を自動生成する `generate_modelfile()` を実装する
  - `llama.cpp` が指定パスに存在しない場合は `git clone` 手順をエラーメッセージに含めて終了する
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

---

- [x] 9. requirements.txt の作成
  - `requirements.txt` にすべての依存パッケージとバージョンを固定して記載する（`transformers`・`trl`・`peft`・`bitsandbytes`・`optuna`・`tensorboard`・`PyYAML` などを含む）
  - `ray[tune]` はオプション依存として別ファイル（`requirements-ray.txt`）に分離する（オプション）
  - _Requirements: 11.2_

---

- [ ] 10. 統合動作確認
- [ ] 10.1 ダミーデータセットを使った学習パイプライン通し確認
  - `data/dummy_train.jsonl` と `data/dummy_eval.jsonl` を使って `train.py` がエラーなく学習ループを完了できることを確認する
  - TensorBoard ログ（`runs/` ディレクトリ）が生成されることを確認する
  - LoRA アダプターが出力ディレクトリに保存されることを確認する
  - _Requirements: 7.4, 9.1_

- [ ] 10.2 GGUF 変換フロー確認
  - `convert_to_gguf.py` がマージ済みモデルから GGUF ファイルと `Modelfile` を生成できることを確認する
  - `llama.cpp` 不在時にエラーメッセージが適切に出力されることを確認する
  - _Requirements: 6.1, 6.6_

---

- [x] 11. (P) `configs/hpo_config.yaml` の作成
  - `configs/hpo_config.yaml` に HPO の設定テンプレート（バックエンド選択・探索空間の範囲・試行数など）を定義する
  - _Requirements: 8.1, 8.6_
