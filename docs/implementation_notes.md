# Diffusion-Jev

Jev風の「型付き判定モデル」（Choice / Noul / Score に確率付きで答えるモデル）を、離散拡散モデルとして作る実装です。

- **拡散モデル化**：既存の自己回帰（AR）モデルを、DiffusionGemma（arXiv:2608.00146）の手順で拡散モデルに変換します。
- **学習データ**：RLCD（arXiv:2307.12950）で選好ペアを作り、それをChoice問題として学習に使います。
- **既定のバックボーン**：`Qwen/Qwen3-VL-4B-Instruct`（4.4B、Apache-2.0、テキスト・画像・動画に対応）

## バックボーンの選び方

DiffusionGemmaの変換は、キャンバス内を**マスクだけで双方向注意にする**ことと、**KVキャッシュを切り詰める**ことに依存しています。そのため、全層が通常の注意機構でできたモデルしか使えません。

| モデル | 可否 | 理由 |
|---|---|---|
| Qwen3-VL-4B / 8B、OneStreamer-4B、VideoChat3-4B、LLaVA-OneVision-2-8B | ✅ | 全層が通常の注意機構で、動画に対応 |
| Gemma 4 12B / 31B | ✅ | スライディングウィンドウ層は層ごとのマスクで対応済み。12Bは16GBのGPUでは学習が難しい |
| Qwen3.5 / 3.8系、MiniCPM-V 4.7、LensVLM、MiMo-V2.6-Distill | ❌ | 線形注意（DeltaNet系）との混成 |
| LFM2.5 / LiquidAI d1 | ❌ | 畳み込み層との混成 |
| Gemma 4 E2B / E4B | ❌ | 層ごとの埋め込み（PLE）のために `input_ids` が必須で、自己条件付けと両立しない |

OneStreamer-4BはQwen3-VL-4Bと同じ構造なので、`--backbone MCG-NJU/OneStreamer-4B` でそのまま差し替えられます。

## 論文との対応

| 論文の記述 | 実装 |
|---|---|
| DG §3：重みを共有した因果エンコーダ（式3/4）と双方向デコーダ（式5） | `djev/model.py` の `encode` / `decode` |
| DG 式6：自己条件付け z = FFW(p̂₀E) | `SelfConditioning`、`DiffusionLM.self_condition` |
| DG Algorithm 1：エントロピー上限付きサンプリング、温度アニーリング、適応停止。既定値はTable 2 | `djev/sampler.py` |
| DG §3.1：ブロック自己回帰（完成したキャンバスをHに追加） | `sampler.generate` |
| DG §3.4：TPF（1回の推論あたりのトークン数）、デノイズのステップ数 | `GenerationResult` |
| DG §4：マルチノミアル拡散（t~U[0,1]、確率tで一様トークンに置換） | `sft.diffusion_loss` |
| DG §8 式13：エンコーダのNTP損失とデコーダのデノイズ損失、50%の自己条件付け、LoRAを全線形層に。既定値はTable 7（PubMedQA） | `djev/sft.py` |
| DG 脚注2：画像・動画は文脈（エンコーダ側）に入れる | `encode(mm=...)`、`schema.prompt_inputs` |
| RLCD §3.1, 付録A：p+ / p- の作り方（Table 9 / 11、それ以前のAssistant行には逆の説明を付ける） | `rlcd.contrastive_prompts` |
| RLCD 付録E：温度1、「Human」で終端、「Assistant」で切り詰め、最大5回まで再サンプリング | `rlcd.sample`、`rlcd.clean_response` |
| RLCD：選好モデルの学習 | 選好ペアをTable 8の形式でChoice問題に変換し、`djev.sft` で学習 |

### 論文から外れている点・論文にない点
- **確率の読み出し**（`decide._readout`）：最終ステップのlogitを、許可された値だけで正規化しています。どちらの論文にも書かれていません。
- **自己条件付けFFWの初期化と学習方法**：最終層をゼロで初期化し、LoRAではなく全体を学習します。論文には書かれていません。
- **画像・動画のプレースホルダートークン**：ノイズとしても予測としても出ないようにしています。
- **SD·RL（DG §5）**：損失の式が論文に書かれていないため、再現していません。
- **RLCDの最後のPPO**：生成モデルを整えるための工程なので、判定モデルには使いません。

## WSL（16GB GPU）での手順

```bash
git clone <this repo> && cd jev_tqa
uv venv -p 3.11 .venv && uv pip install -p .venv torch transformers peft accelerate pytest av
.venv/bin/python -m pytest -q tests

# 1) RLCDで選好データを作る（入力はAnthropic HHの会話プレフィックス）
.venv/bin/python -m djev.rlcd --prompts data/hh_prefixes.jsonl --task harmless --out data/rlcd_harmless.jsonl

# 2) 拡散SFT（LoRA。16GBならbatch 1と勾配チェックポイントを推奨）
.venv/bin/python -m djev.sft --data data/rlcd_harmless.jsonl --out runs/djev-qwen3vl4b \
    --batch 1 --grad-checkpointing

# 3) 判定する
.venv/bin/python -m djev.decide --checkpoint runs/djev-qwen3vl4b --request examples/request.json
```

### 学習データの形式（JSONL、1行に1件）

```json
{"state": "...", "media": [{"type": "video", "path": "clip.mp4"}],
 "questions": [{"type": "noul", "id": "urgent", "question": "Is it urgent?"},
               {"type": "choice", "id": "route", "question": "Who handles it?", "options": ["bot", "human"]},
               {"type": "score", "id": "anger", "question": "How angry?", "levels": ["calm", "annoyed", "angry"]}],
 "answers": {"urgent": "true", "route": "2", "anger": "3"}}
```

答えの値の決まり：Choiceは選択肢の番号（1〜n）、Noulは `true` / `false`、Scoreはレベルの番号（1〜n）。選択肢やレベルは最大9個までです。
