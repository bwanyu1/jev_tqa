# 実際の最終AR・拡散Jevの回答過程

2026年10月25日、RTX 4060 Ti 16GB。同じ英語の問い合わせと8つのはい／いいえ質問を、最終AR `runs/ar_sft_v5` と最終拡散Jev `runs/jev_infill` に渡した。追加学習なし、seed=20261025。

![実測ログの比較](jev_comparison.gif)

## 実際に何が起きたか

ARは固定された回答テンプレートの穴を左から1つずつ埋め、8問で8回のforward。最初のforwardにはプロンプトのprefillを含む。JSONの記号やキーを自由文として1トークンずつ生成した記録ではない。

拡散は入力を1回エンコードした後、回答キャンバス128トークンのうち8つの穴へ乱数を入れて開始し、3回のdecodeで停止した。GIFで示す更新3回とは別に、入力エンコードが1回ある。本例では8項目の最大候補は初回から停止時まで同じだった。入力穴の乱数・自己条件付け・候補内確率は更新されている。最初から誤りなく読めた単一の例であり、反復による正解率改善を示すものではない。

両モデルの実際の返り値は、二重請求=true、返金要求=true、今日の対応=true、サービス停止=false、解約希望=false、メール連絡=true、請求の問題=true、住所変更=false。

## GIFの読み方

- AR側のカードは、各forwardで実際に選んだ答えを順に表示する。
- 拡散側の「入力」は、そのdecodeへ入った各穴の実際のトークン。長い文字列は表示を短縮し、非ASCII文字はエスケープする。元の文字列とIDは生ログに保持。
- 拡散側の大きな「はい／いいえ」と確率は、その時点のlogitを許された候補で正規化した最大予測。ノイズ入力そのものや、取り消せない確定値ではない。
- 黄色は新しく追加した、または前回から最大候補が変わったカード。確率だけ変わった場合は黄色にしない。
- GIFは1ステップごとに一定の表示時間を割り当てた。**実時間の速度比較ではない。** 入力エンコードや各ステップの計算量も両方式で異なる。

**最終Jevの現行APIは初回予測を返す。** 反復の最後の値・確率をAPI出力と読み替えない。既存の動画試用ではfinalが悪化した例もあり、並列更新や確率の上昇が正しさを保証するわけではない。

## 元ログと再現

- [ARの生トレース](ar_trace.json)
- [拡散Jevの生トレース](diffusion_trace.json)
- [ARだけのGIF](ar_process.gif) / [拡散JevだけのGIF](diffusion_jev_process.gif)
- [初期状態](jev_comparison_initial.png) / [最終表示](jev_comparison_final.png)

`record_generation_process.py` はARのforward hookと拡散のdecodeラッパーで観測する。サンプラー・重み・最大48反復の停止規則は変更しない。ARの記録値が実際の返り値と一致し、拡散の初回記録が実際のAPI読み出しと一致することをスクリプト内で検証した。記録処理にGPU→CPUの転送等が加わるため、所要時間は性能測定に使わない。

GPUサーバーの元のチェックポイントがある環境で、リポジトリ直下から実行する例：

```bash
python -m scripts.record_generation_process --mode diffusion --out docs/generation/diffusion_trace.json
python -m scripts.record_generation_process --mode ar --out docs/generation/ar_trace.json
```

GIFの再描画（既定の日本語フォントパスはmacOS用）：

```bash
python -m scripts.render_jev_process_gif \
  --ar docs/generation/ar_trace.json \
  --diffusion docs/generation/diffusion_trace.json
```

これは最終の判定モデル同士の過程。汎用拡散v3の自由文生成を撮ったGIFではない。旧 `docs/compare_*.gif` は別の過去実験であり、今回の最終モデルの過程として再利用していない。
