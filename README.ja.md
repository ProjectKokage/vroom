# vroom

[English](README.md) | **日本語**

HY-Motion 1.0 は、テキストから人体動作を生成します。
VRM アプリケーションがその出力をリターゲットするには、ヒューマノイドボーンの対応関係を持つ可搬ファイルが必要です。
`vroom` は、身体の回転とルートの平行移動を、アバターに依存しない [VRM Animation 1.0](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)（`.vrma`）ファイルへ書き出します。

VRMA 対応プレイヤーは、生成したクリップを VRM ヒューマノイドへ適用できます。
`vroom` が担うのは、アニメーションファイルの作成と検証です。
描画とプレビューは行いません。

## コマンド

| コマンド | 用途 |
| --- | --- |
| `generate` | 英語のモーションプロンプトから VRMA クリップを一つ生成します |
| `curate` | サンプリング済みの LINEAR 補間 VRMA から、範囲を限定した回転専用クリップを作ります |
| `inspect` | VRM または VRMA の GLB を要約します |
| `validate` | 一つ以上の VRMA ファイル、または ZIP 内の全 VRMA ファイルを検証します |

GLB、VRM、VRMA を扱うコードは Python 標準ライブラリだけを使い、実行時依存パッケージを持ちません。
基本パッケージだけで `curate`、`inspect`、`validate` を実行できます。
`generate` に必要な Torch と HY-Motion の推論処理は、オプション依存グループ `inference` に分離しています。
生成処理は Python プロセス内で完結し、GLB データを直接書き出します。

## 出力するデータ

| データ | 生成する VRMA の内容 |
| --- | --- |
| ヒューマノイド階層 | リターゲットに使う、自己完結した VRM の T ポーズノード |
| ルートモーション | `hips` のローカル平行移動 |
| 身体動作 | 公開済み HY-Motion チェックポイントが出力する 22 関節のローカルクォータニオントラック |
| 指 | 公開済みチェックポイントが指の回転を生成しないため省略 |
| 表情と視線 | 表情アニメーションと LookAt アニメーションは生成しません |

## クイックスタート

### 必要なソフトウェア

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash、Git、[Git LFS](https://git-lfs.com/)

以下のコマンドは、Git からチェックアウトしたリポジトリと Bash を前提とします。

### HY-Motion のライセンス

`vroom` の wheel とソース配布物には、HY-Motion のコードとチェックポイントを含めていません。
HY-Motion のコード、チェックポイント、出力には、地域と用途に関する制限を含む [Tencent HY-Motion 1.0 Community License](https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/License.txt) が適用されます。
HY-Motion をダウンロードまたは実行する前に、このライセンスを確認してください。

### インストールと生成

HY-Motion のサブモジュールを初期化し、Git LFS が管理する実行用アセットを取得してから、ロック済みの推論環境をインストールします。

```bash
./scripts/bootstrap_hy_motion.sh
uv sync --extra inference
```

ブートストラップスクリプトが取得するのは、実行時に使う統計ファイルとスケルトンアセットです。
モーション生成用チェックポイントは取得しません。
フル版のチェックポイントをダウンロードします。

```bash
uv run --extra inference huggingface-cli download \
  tencent/HY-Motion-1.0 \
  --include "HY-Motion-1.0/*" \
  --local-dir vendor/HY-Motion-1.0/ckpts/tencent
```

クリップを生成し、出力を検証します。

```bash
uv run --extra inference vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

`generate` は、デフォルトパスのフル版チェックポイント、4 秒の動作時間、シード `42`、デバイスの自動選択を使います。
出力先のディレクトリがなければ、コマンドが作成します。
HY-Motion のチェックアウト内に二つのテキストエンコーダーがそろっていない場合、初回生成時に Qwen3-8B と CLIP-L を Hugging Face からダウンロードし、Hugging Face のキャッシュへ保存します。

## インストール先とモデルの配置

### Lite 版のモーションチェックポイント

Lite 版もフル版と同じコマンドでダウンロードできます。
`HY-Motion-1.0/*` を `HY-Motion-1.0-Lite/*` に置き換えてください。
どちらのモデルも推論には大きなメモリが必要になるため、両チェックポイントのハードウェア要件は [HY-Motion のモデル一覧](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo) で確認してください。

### wheel からのインストール

wheel には `vroom` が入りますが、HY-Motion の実行環境は入りません。
`inference` を含めて wheel をインストールします。

```bash
python3.12 -m pip install \
  "/absolute/path/to/vroom-0.1.0-py3-none-any.whl[inference]"
```

HY-Motion は別のディレクトリへクローンし、[公式のセットアップ手順](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#get-started-with-hy-motion-10) に従って Git LFS アセットとモーション生成用チェックポイントを取得します。
そのうえで、HY-Motion のルートディレクトリとチェックポイントの場所を指定します。

```bash
export VROOM_HY_MOTION_ROOT=/absolute/path/to/HY-Motion-1.0
export HY_MOTION_MODEL_PATH=/absolute/path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0
```

コマンドごとに `--hy-motion-root` と `--model-path` を渡しても構いません。

### テキストエンコーダーのローカル配置

`--text-models auto` は、Qwen3-8B と CLIP-L のローカルディレクトリが両方あれば、それらを使います。
どちらかがなければ、初回生成時に HY-Motion が Hugging Face からテキストエンコーダーをダウンロードします。

`auto` が検出する場所へ、二つのエンコーダーをダウンロードします。

```bash
uv run --extra inference huggingface-cli download \
  Qwen/Qwen3-8B \
  --local-dir vendor/HY-Motion-1.0/ckpts/Qwen3-8B

uv run --extra inference huggingface-cli download \
  openai/clip-vit-large-patch14 \
  model.safetensors \
  config.json \
  merges.txt \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  --local-dir vendor/HY-Motion-1.0/ckpts/clip-vit-large-patch14
```

`--text-models local` を指定するとローカルディレクトリが必須になり、どちらかがなければ推論前にエラーになります。
モデルの重みは Git の追跡対象外です。
`inference` には、大きな重みファイルの転送に使う Hugging Face の Xet ヘルパーも含まれます。

## モーションの生成

指定できる動作時間は、選択したチェックポイントが決めます。
公開済みチェックポイントは約 0.667 秒から 12 秒までを受け付け、30 fps のモーションを生成します。
チェックポイントの範囲から外れた動作時間は、`vroom` がエラーにします。

HY-Motion は、60 語未満の英語プロンプトを推奨しています。
外見、背景、カメラワークを避け、体幹と四肢の動きを記述してください。
メモリ消費を抑える場合、HY-Motion は一つのシード、30 語未満のプロンプト、5 秒未満の動作を推奨しています。

出力を再現したい場合やデバイスを固定したい場合は、引数を指定します。

```bash
uv run --extra inference vroom generate \
  "A person crouches, stands, and raises both arms." \
  --duration 3 \
  --seed 7 \
  --device mps \
  --output output/raise-arms.vrma
```

CFG スケール、ODE 推論ステップ数、テキストエンコーダーの配置、モデルパス、接地処理を含む全引数は、`uv run --extra inference vroom generate --help` で確認できます。

一つのプロンプトから、シードの異なる再現可能な結果を複数作る場合、Python から `HyMotionBackend.generate_variants` を呼び出せます。
このメソッドは、重複しない 1 個から 16 個のシードを受け取り、最大 16 件からなるプロンプトのバッチを一度エンコードして、一回のモデルロード後に全シードを生成します。
単一シード用の API は `generate` です。

### デバイスの選択

| 対象 | 引数 | 動作 |
| --- | --- | --- |
| 自動選択 | `--device auto` | CUDA、MPS、CPU の順に選びます |
| Apple Silicon | `--device mps` | 拡散モデルを MPS、テキストエンコーダーを CPU で実行します |
| Apple Silicon ですべて MPS | `--device mps --text-device same` | テキストエンコーダーも MPS での実行を試みます |
| NVIDIA GPU | `--device cuda:0` | 最初の CUDA デバイスを使います |
| CPU | `--device cpu` | 拡散モデルとテキストエンコーダーを CPU で実行します |

HY-Motion のデバイス補助処理は CUDA または CPU を選びます。
`vroom` は MPS の検出を追加し、MPS 用ノイズを CPU シードから作り、PyTorch の MPS CPU フォールバックを有効にします。
Qwen エンコーダーはデフォルトで CPU に配置します。
実際の生成速度は、使用する PyTorch と macOS のバージョンによって変わります。

`vroom` が MPS フォールバックを有効にする前に、組み込み先のアプリケーションが Torch をインポートすることがあります。
その場合は、プロセスの起動前に `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定してください。

PyPI から取得する固定バージョンの Torch は、Linux x86-64 で CUDA の依存パッケージを導入します。
リポジトリに含まれるロックは、Windows の CUDA 環境を定義していません。

## モーションのキュレーション

`curate` は、構造を検証できるサンプリング済みの LINEAR 補間 VRMA を受け取り、決定論的な回転専用クリップを書き出します。

```bash
uv run vroom curate output/wave.vrma \
  --output output/wave-upper-body.vrma \
  --bones spine chest upperChest neck head leftUpperArm rightUpperArm \
  --duration 2.0 \
  --fade-in 0.25 \
  --fade-out 0.25 \
  --rotation-gain 1.25
```

このコマンドは、ルートの平行移動と指定外のトラックを削除します。
選択した回転を `inverse(first) * frame` で表し、クォータニオンの符号を最短経路上で連続させたうえで、残したサンプルの時間を変更します。
Smoothstep フェードにより、最初と最後のキーフレームは恒等回転になります。

増幅したトラックやフェードしていないトラックでは、巻き込みを含む回転から恒等回転へ戻る連続なクォータニオン経路を作れない場合があります。
その場合、コマンドは出力を書き出さずに失敗し、より長いフェードアウトまたは狭いソース範囲を求めます。

| 引数 | 効果 |
| --- | --- |
| `--bones` | 指定したヒューマノイド回転トラックを残します |
| `--start-frame`、`--end-frame` | 終了フレームを含むソース範囲を選びます |
| `--duration` | サンプル間の相対時間を保ったまま、選択したサンプルの時間を変更します |
| `--fade-in`、`--fade-out` | 始点と終点に Smoothstep フェードを適用します |
| `--rotation-gain` | 先頭フレームを基準にした回転を `0` から `4` の範囲で拡大または縮小し、デフォルト値には `1` を使います |
| `--rotation-projection` | `none` は回転全体を保ち、`twist-x`、`twist-y`、`twist-z` はローカル軸まわりのねじりだけを残します |
| `--name` | 出力アニメーション名を設定します |

回転ゲインは、フェードより前にクォータニオン補間を使って適用します。
`1` より大きい値は、小さな動きを増幅します。
ねじりへの射影には Swing-Twist 分解を使うため、`--rotation-projection twist-y` を指定すると、オイラー角へ変換せずに頭と首のトラックをヨー回転だけにできます。

出力には、リターゲットに必要なヒューマノイドノードを残します。
ソースの SHA-256、バイト長、変換設定は `asset.extras.vroomCuration` に記録します。

### キュレーションの上限

`curate` は、次の上限を超えた入力を、出力の書き込み前に拒否します。

| 対象 | 上限 |
| --- | ---: |
| ソースファイル | 64 MiB |
| ノード | 512 |
| アクセサごとのサンプル | 18,000 |
| デコードする数値アクセサ成分の合計 | 1,000,000 |
| アニメーションのチャンネルまたはサンプラー | 256 |
| キュレーション後の長さ | 3,600 秒 |
| シリアライズ後の出力 | 16 MiB |

## ファイルの確認と検証

```bash
uv run vroom inspect path/to/avatar.vrm
uv run vroom inspect output/wave.vrma --json
uv run vroom validate output/wave.vrma
uv run vroom validate path/to/motion-pack.zip
```

バリデーターは、GLB コンテナ、アクセサ、キーフレーム時刻、クォータニオンの正規化、ヒューマノイドの対応関係、`vroom` が実装するヒューマノイドアニメーションの制約を検査します。
ヒューマノイドボーンはスケールトラックを持てず、平行移動トラックを持てるボーンは `hips` だけです。

VRM 0.x アバターを読み込むときは、旧仕様の親指ボーン名を VRM 1.0 相当の名前へ対応付けます。

## HY-Motion から VRMA への変換

HY-Motion は、各回転を回転行列の二つの列を交互に並べた形式で保持します。
`vroom` は HY-Motion と同じ Gram-Schmidt 変換を適用し、glTF の `[x, y, z, w]` 順のクォータニオンへ変換したうえで、時間軸上の符号を連続させます。

HY-Motion の木製マネキンに含まれるスケルトンは、VRM 1.0 と同じ座標系を使っています。
この座標系は右手系で、単位はメートル、上方向は +Y、前方は +Z、解剖学上の左側は +X です。
VRMA の身体動作に必要なデータは回転とルートの平行移動なので、`vroom` は推論時にメモリ消費の大きいメッシュスキニングを無効にします。
続いて、順運動学によってソースのレストポーズに含まれる床面オフセットを考慮し、生成したモーションを接地します。
最後に、各キーフレームにおける `hips` のローカル平行移動を書き出します。

## 開発

短時間で完了するテストスイートは、Torch とモデルの重みを必要としません。

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest
```

モデルをダウンロードした後は、オプトイン形式の推論スモークテストを実行できます。

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
  uv run --extra inference pytest -m integration
```

エンドツーエンドの生成テストでアクセラレーターを指定するには、`HY_MOTION_TEST_DEVICE=mps` または `HY_MOTION_TEST_DEVICE=cuda:0` を設定します。

`AliciaSolid_vrm-0.51.vrm` と `VRMA_MotionPack.zip` は、再配布に制限のあるローカルテストフィクスチャです。
Git は両ファイルを追跡せず、配布パッケージにも含めません。
ファイルがなければ、対応するテストをスキップします。
