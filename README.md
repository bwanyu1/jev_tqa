# Diffusion-Jev

既存の視覚言語モデル **Qwen3-VL-4B-Instruct** を拡散モデルに変換し、さらに複数の判定を「回答テンプレートの穴埋め」として学習した実験プロジェクトです。

この非公開リポジトリには、実装・最終2種類のモデル・共通のベースモデル・実際に試した入力と出力・検証レポートをまとめています。

## 入っているモデル

| モデル | 保存先 | 用途 |
| --- | --- | --- |
| 最終の拡散Jev | `models/final_diffusion/jev/` | Choice・はい/いいえ・段階評価を、候補内確率付きでまとめて返す |
| 拡散のみ（汎用拡散v3） | `models/final_diffusion/diffusion_only/` | Jev化前の文章生成を試す |
| 共通の前段重み | `models/final_diffusion/dependencies/` | v2と初期の汎用拡散。差分の適用に必要 |
| 共通のQwen本体 | `models/final_diffusion/backbone/` | 両モデルが使用するベースモデル |

重みは **LoRA差分＋自己条件付けFFW** です。最終差分だけでは動作しません。用意したローダーが共通本体と前段差分を順に読み込みます。`models/final_diffusion/runs/` の相対シンボリックリンクも必要です。

**最終AR比較モデルの重みは、この手元の保存物には含まれません。** ARとの既存の比較結果・ログはレポートに記録しています。

## 取得とセットアップ

Python 3.11を使用。学習・これまでのGPU試用はRTX 4060 Ti 16GBで実施しました。CUDA環境を推奨します。MacのMPSやCPUは指定できますが、同じ速度や動作を確認した環境ではありません。

### 1. Git LFS付きでcloneする

GitHubへのアクセス権が必要です。Git LFSをインストールしてから実行します。

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/bwanyu1/jev_tqa.git
cd jev_tqa
git lfs pull
```

モデル・前段・本体は合計約10.3GiB。Qwen本体の大きな2ファイルは **1GiB単位のバイナリ断片** としてLFSに保存しています。元のバイト列へ復元するため、PyTorchの導入前に次を実行してください。Python標準ライブラリだけで動きます。

```bash
python3 scripts/restore_model_files.py --verify
```

断片と復元後のSHA-256を照合し、最後に取得元と同じ46ファイルを検証します。LFSのポインタだけでは復元できません。復元済みのファイルはGit管理から除外します。clone後はLFS内部のキャッシュ・作業ツリーの断片・復元した本体が並ぶので、**空き容量は少なくとも35GiB程度** を用意してください。

GitHubのLFSには保存・転送枠があります。利用環境の上限は [GitHub公式のLFS説明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) を参照してください。

### 2. Python環境を作る

`uv`を使う例です。依存バージョンは実際の試用環境を `requirements.txt` に記載しています。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q tests
```

CUDA用PyTorchの配布先が必要な環境では、環境に適したPyTorchを先に導入してください。以降のコマンドは **リポジトリ直下** で実行します。

## 最終拡散Jevを使う

入力例は `examples/request.json`。文章と複数の問いを渡します。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m scripts.run_local \
  --model jev --device cuda --request examples/request.json
```

返り値は `decisions` に各問の `answer` と候補内の `probs`、`steps` に反復回数を持つJSONです。例は返り値の構造を示したもので、実測値を作ったものではありません。

```text
{
  "decisions": {
    "urgent": {"answer": "true", "probs": {"true": ..., "false": ...}},
    "route": {"answer": "2", "probs": {"1": ..., "2": ..., "3": ...}}
  },
  "steps": ...,
  "readout": "first",
  "load_seconds": ...,
  "processing_seconds": ...
}
```

`load_seconds` はモデル読み込み、`processing_seconds` は入力処理から反復終了までの時間です。初回のGPUウォームアップを自動的に取り除くわけではありません。既存レポートの定常時の時間と同一条件とは限りません。

### 入力の形式

```json
{
  "state": "My invoice was charged twice. Please refund the duplicate charge.",
  "questions": [
    {"type": "noul", "id": "refund", "question": "Is a refund requested?"},
    {"type": "choice", "id": "team", "question": "Which team should handle this?", "options": ["billing", "delivery", "technical"]}
  ]
}
```

Choiceは1から始まる候補番号、Noulは `true` / `false`、Scoreは1から始まる段階番号です。APIの回答値は文字列です。候補は2〜9個。最終Jevは回答枠128トークンなので、項目IDを短くし、超える場合は問いを分けてください。候補内確率は、現場での正答確率を保証する値ではありません。

画像や動画は、同じ入力に次を追加します。

```json
"media": [{"type": "image", "path": "docs/visual_practical/images/ui_success.png"}]
```

複数画像は `media` に順番に列挙します。動画は `type: "video"` とMP4のパスを指定。新しいCLIではPyAVによる動画デコードを使い、2fps・最大32フレームで処理します。動画に関する実務上の制限は下記の検証結果を参照してください。

### Pythonから読み込む

```python
from models.final_diffusion.load_models import load_model

model, tokenizer, processor = load_model("jev", device="cuda")
```

この関数はロード中に作業ディレクトリを変更するため、初期化時に直列で呼びます。Jevの実際の穴埋め処理は `scripts/run_local.py` と `djev/infill.py` を参照してください。

## 拡散のみのモデルで文章生成する

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m scripts.run_local \
  --model diffusion_only --device cuda \
  --prompt 'Explain why the sky is blue in two sentences.'
```

汎用拡散v3はJevの判定学習前のモデルです。生成品質、日本語、繰り返し、少ない反復での品質には課題が残っています。既定はキャンバス256、最大192反復、最大2キャンバスです。最終Jevの多問判定の速度を、そのまま自由文生成に当てはめないでください。

## これまでに分かったこと

- **多問判定**：16問の条件では拡散0.55秒／AR1.13秒。全体の正解率はARが上でした。
- **実務**：問い合わせや画面を一部読めても、送信・返品受付・合格判断を間違える例があります。
- **複数画像**：2枚照合に手応えがありますが、類似物の6枚一括選択は不安定でした。
- **動画**：大まかな状態変化は拾えましたが、逆再生の方向・不明判定に課題。詳細な動作定義だけでは解決せず、今回の動画ではARが速い結果でした。

数値は条件ごとの観測です。モデル起動時間は除外し、初回呼び出しは別に扱っています。実務・画像・動画の手動試用は少数例で、一般的な正解率や実運用品質を示すものではありません。

| 読むもの | 内容 |
| --- | --- |
| [最終比較レポート](docs/Diffusion-Jev_report.md) | モデル・手法・最終結果 |
| [手法と学習データ](docs/Diffusion-Jev_methods.md) | 拡散化・Jev化の詳細 |
| [実務の課題](docs/Diffusion-Jev_operational_findings.md) | 業務規則・保留・項目間の矛盾 |
| [画面と点検画像](docs/Diffusion-Jev_visual_practical.md) | 実際の画像と出力 |
| [複数画像](docs/Diffusion-Jev_multi_image_practical.md) | 参照照合・分割・一括 |
| [動画](docs/Diffusion-Jev_video_practical.md) | 正順・逆順・静止画と処理時間 |
| [作業動作の再定義](docs/Diffusion-Jev_video_phase_definitions.md) | 名称のみ／詳細定義の比較 |
| [今後の課題](docs/Diffusion-Jev_future_work.md) | 定義・学習・順次入力、日本語、SD·RL |
| [発表スライドPDF](docs/presentation/Diffusion-Jev_slides.pdf) | 本編42枚＋補足13枚 |
| [開発記録](docs/DEVELOPMENT_LOG.md) | 実施内容の履歴 |

## ディレクトリ

```text
djev/                      拡散・判定・学習・評価の実装
scripts/                   試用・データ準備・復元・起動
models/final_diffusion/    最終Jev、拡散のみ、共通の重み
examples/                  入力例と試用リクエスト
logs/                      動画試用などの生出力
docs/                      レポート、画像、動画、各試用の生出力
tests/                     既存のユニットテスト
runs/toy/                  小さな動作確認用チェックポイント
```

`.venv`、キャッシュ、作業用 `tmp`、認証情報は管理対象外です。学習用データ一式や最終AR重みは手元にないため含みません。古い実験スクリプトにはGPUサーバーのパスや環境を前提としたものがあります。配布した重みを手元で使う入口は `scripts/run_local.py` です。

## 検証範囲と出典

新しい配布用CLIは引数・構文を確認し、既存テストは19件成功。別ディレクトリへのclone、Git LFS経由のJev重み1ファイルの取得・SHA-256一致、全31個のLFSオブジェクトの取得可能状態も確認しました。モデルの元ファイル46件は取得元とのSHA-256を照合済み。本体1ファイルを新規に復元し、取得元と同じSHA-256になることも確認しました。新しいCLIでのGPU推論と、Macでの実モデル推論はまだ実施していません。記載した性能値は既存のGPU試用によるものです。

ベースモデルは [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、固定revisionは `ebb281ec70b05090aa6165b016eac8ec08e71b17`。モデル本体の説明とライセンスは同梱の [モデルカード](models/final_diffusion/backbone/README.md) を参照してください。動画の出典は [lerobot/aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee) で、固定revisionと元パスを [provenance.json](docs/video_practical/source/provenance.json) に保存しています。

初期実装の論文対応・RLCDなどの説明は [初期実装メモ](docs/implementation_notes.md) に保存しました。最終モデルの実行には上記の手順を使ってください。
