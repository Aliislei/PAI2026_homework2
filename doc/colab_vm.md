# Colab VM の手順

Colab の VM は手元から `colab` CLI で操作する。Drive とのやり取りは VM 上の rclone で行う。

- **保存先**: 環境変数 `RCLONE_REMOTE_ROOT` で rclone のリモートとフォルダを指定する。例: `remote:pai_homework2`
- **VM に送る認証情報**: `rclone.conf` と Hugging Face トークンのファイル

認証情報はリポジトリに入れない。

`google/paligemma-3b-pt-224` の利用申請は先に済ませておく。

## 実行

```bash
colab new --gpu G4 -s pai

colab upload -s pai <rclone.conf のパス> /content/rclone.conf
colab upload -s pai <トークンファイルのパス> /content/hf_token
colab upload -s pai scripts/rollout.py /content/rollout.py

colab exec -s pai --timeout 3600 -f notebooks/collect_and_analyze.ipynb
```

結果は `approach_analysis/results.tar.gz` として保存先に置かれる。手元に回収する場合は次のコマンドを使う。

```bash
rclone copyto $RCLONE_REMOTE_ROOT/approach_analysis/results.tar.gz results/results.tar.gz -P
```

## 停止

使い終わったら必ず VM を止める。止めるまでコンピュートユニットが消費される。

```bash
colab stop -s pai
colab sessions
```

## 注意点

- `colab exec` は無出力が続くと手元側がタイムアウトする。カーネル側の処理は続く
- `colab exec` はセルが失敗しても次へ進み、終了コード 0 を返す。結果はログのエラーで判定する。失敗しても最後のセルが認証ファイルを消すので、再実行では `rclone.conf` とトークンをアップロードし直す
- `drive.mount` は CLI では使わない
- proxy token の寿命は 1 時間で、長い実行では VM が回収されることがある。回収されても、Drive のアーカイブから続きを実行できる
- 小さなファイルを大量に転送しない。結果は 1 個のアーカイブにまとめる
- `colab exec` は長時間かかるので、手元ではバックグラウンドで実行する
