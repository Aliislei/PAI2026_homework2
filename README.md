# pi0.5 アプローチの推論高さ検証

Physical AI 応用編 1 最終課題の、受講者個人の取り組みです。講座教材ではありません。

深度画像を持たない VLA モデル pi0.5 の Libero チューニング済みモデルを対象に、対象物へのアプローチ中の推論が距離をどれだけ正しく捉えているかを測定し、推論を固定する対策を試しました。

## 結果

1000 ロールアウト、9,321 推論を分析しました。

- 計画の精度は距離に対して単調ではありません。14〜20cm が最も安定し、8〜12cm で突っ込み側の裾が最大になります（p90 +7.4〜+8.7cm）
- 誤差の大きいロールアウトでは、手先が止まった後も下降指令が続きます
- 予定軌道の最下点までの距離で推論を固定する対策は、タスクによって効果が分かれました

| 層 | 本数 | 過剰下降ステップ数 | p 値 |
|---|---:|---:|---:|
| 机上の物体タスク（5 タスク） | 46 | 9.17 → 6.07 | 0.07 |
| 棚のタスク | 33 | 5.55 → 7.15（悪化） | 0.05 |
| 過剰下降の少ないタスク | 121 | 1.29 → 1.41 | 0.36 |

過剰下降ステップ数は、下降指令中に高さが動かなかったステップ数の平均です。

## 構成

| パス | 内容 |
|---|---|
| `scripts/rollout.py` | モデル、環境、ロールアウト、対策の共通処理 |
| `notebooks/collect_and_analyze.ipynb` | データ収集、分析、問題ロールアウトの抽出 |
| `notebooks/atomic_compare.ipynb` | 対策の比較 |
| `notebooks/render_videos.ipynb` | 検証動画の作成 |
| `doc/theme.md` | テーマ、仮説、検証方法 |
| `doc/report.html` | レポートの原稿 |
| `doc/colab_vm.md` | Colab VM の手順 |

## 実行

Google Colab の G4 ランタイムで実行します。VM の手順は `doc/colab_vm.md` を参照してください。

1. `notebooks/collect_and_analyze.ipynb` で、対策なしのデータ収集と分析を行う
2. `notebooks/atomic_compare.ipynb` で、対策ありの比較を行う
3. `notebooks/render_videos.ipynb` で、動画を作る

保存先は、環境変数 `RCLONE_REMOTE_ROOT` で rclone のリモートとフォルダを指定してください。

## 使用モデルと環境

- モデル: `lerobot/pi05_libero_finetuned_v044`（推論のみ。学習はしていません）
- 環境: Libero（`libero_spatial`、`libero_object`）
- ライブラリ: `lerobot[pi,libero]==0.6.1`
