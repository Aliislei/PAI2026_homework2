# Colab VM の手順

Colab の VM は手元から `colab` CLI で操作する。Drive とのやり取りは VM 上の rclone で行う。

- **Drive の保存先**: `remote:pai_homework2/`
- **VM に送る認証情報**: `~/.config/rclone/rclone.conf`（`remote:` だけを含む）
- **Hugging Face トークン**: Drive の Google ドキュメント `pai_homework2/hf_token` に置く。ノートブックがテキストとして書き出し、VM へ取得する

`rclone.conf` はトークンを含むため、リポジトリに入れない。Google Drive を再認証したときは作り直す。

`google/paligemma-3b-pt-224` の利用申請は先に済ませておく。

## 実行

```bash
colab new --gpu G4 -s pai

colab upload -s pai ~/.config/rclone/rclone.conf /content/rclone.conf

colab exec -s pai --timeout 3600 -f notebooks/collect_and_analyze.ipynb
```

結果は Drive の `pai_homework2/approach_analysis/results.tar.gz` に保存される。手元に回収する場合は次のコマンドを使う。

```bash
rclone copyto remote:pai_homework2/approach_analysis/results.tar.gz results/results.tar.gz -P
```

## 停止

使い終わったら必ず VM を止める。止めるまでコンピュートユニットが消費される。

```bash
colab stop -s pai
colab sessions
```

## 注意点

- `colab exec` は無出力が続くと手元側がタイムアウトする。カーネル側の処理は続く
- `drive.mount` は CLI では使わない
- proxy token の寿命は 1 時間で、長い実行では VM が回収されることがある。回収されても、Drive のアーカイブから続きを実行できる
- 小さなファイルを大量に転送しない。結果は 1 個のアーカイブにまとめる
- `colab exec` は長時間かかるので、手元ではバックグラウンドで実行する
