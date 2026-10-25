# 最終モデルのローカル保存

GPUサーバー `root@yuuya-1:/root/jev_tqa` から取得。

| ディレクトリ | 内容 |
| --- | --- |
| `jev/` | 最終拡散Jev（元の `runs/jev_infill`） |
| `diffusion_only/` | Jev化前の汎用拡散モデルv3（元の `runs/general_v3`） |
| `dependencies/general_v2/` | 両モデルの初期化に必要な前段 |
| `dependencies/general_12000/` | 両モデルの初期化に必要な前段 |
| `backbone/` | 共通のQwen3-VL-4B-Instruct本体・プロセッサ |
| `runs/` | 元の設定ファイルの相対参照を解決するローカルシンボリックリンク |

モデルはLoRA差分と自己条件付け重みで保存されているため、`jev/` や `diffusion_only/` だけを取り出すと完結しない。コピーする際はこの親ディレクトリ全体を扱う。学習途中の `step_*` は取得対象外。最終の重みと設定ファイルは変更していない。

## 読み込み

このリポジトリのPython環境で、リポジトリ直下から実行する。

```python
from models.final_diffusion.load_models import load_model

# どちらか一方ずつ読み込む
model, tokenizer, processor = load_model("jev")
# model, tokenizer, processor = load_model("diffusion_only")
```

GPUを明示する場合は `device="cuda"` などを渡す。`load_model()` は共通のローカル本体を使い、元の前段チェックポイントを順番に適用する。内部ではロード中だけ作業ディレクトリを変更するので、初期化時に直列で呼ぶ。

本体revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`。
取得後のファイル照合結果は `download_manifest.json` に保存。


## GitHubから取得した場合

大きな本体ファイル2個は1GiB単位のGit LFS断片で格納する。リポジトリ直下で `git lfs pull` の後に `python3 scripts/restore_model_files.py --verify` を実行して復元する。判定・文章生成のコマンドと依存環境は[ルートREADME](../../README.md)を参照。
