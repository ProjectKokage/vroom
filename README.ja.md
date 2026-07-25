# vroom

[English](README.md) | **日本語**

HY-Motion はテキストから人体動作を生成しますが、VRM アバター間で再利用できるアニメーションファイルは出力しません。
`vroom` はその隔たりを埋め、英語で記述した人体動作から、アバターに依存しない VRM Animation 1.0（`.vrma`）ファイルを生成します。

動作の生成を担うのは HY-Motion 1.0 です。
`vroom` は生成結果を、自己完結した VRM ヒューマノイドの T ポーズ階層を持つ VRMA ファイルへ変換します。
VRMA 対応プレイヤーは、このファイルをアバターへリターゲティングできます。
アニメーションの描画やプレビューは行いません。

GLB、VRM、VRMA を扱う部分は、Python 標準ライブラリだけで動作します。
Torch と HY-Motion の推論処理は、オプション依存グループ `inference` に分離しています。
生成処理では、FBX SDK、Blender、JavaScript エクスポーター、サードパーティー製の Python VRM ライブラリを使いません。

## 機能

- GLB 2.0 コンテナを読み書きし、バイナリアクセサをデコードします。
- VRM 0.x と VRM 1.0 のヒューマノイドメタデータを読み取ります。
- VRM Animation 1.0 を書き出し、メタデータとバイナリアクセサについて、実装対象の制約を検証します。
- HY-Motion の 6 次元回転表現を変換し、SMPL-H のボーンを VRM のボーンへ対応付けます。
- CUDA、MPS、CPU を自動選択するか、使用するデバイスを明示できます。
- `generate`、`inspect`、`validate` の各コマンドを CLI から実行できます。

生成されるファイルには、VRM ヒューマノイドの T ポーズ階層と `hips` ボーンの平行移動が入ります。
`vroom` は、公開済み HY-Motion チェックポイントが出力する指を除く全身 22 個のボーンについて、ローカル回転をクォータニオントラックとして収録します。
公開済みチェックポイントは指の回転を生成しないため、指のボーンは含めません。
表情と視線のアニメーションも生成しません。

## セットアップ

### 必要なソフトウェア

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash、Git、[Git LFS](https://git-lfs.com/)

特記のない限り、以下のコマンドは Git からチェックアウトしたリポジトリと Bash を前提とします。
`inspect` と `validate` は基本パッケージだけで動作しますが、`generate` には `inference` の依存パッケージ、HY-Motion 本体、モデルの重みが必要です。

### HY-Motion のライセンス

HY-Motion は外部ランタイムであり、`vroom` の wheel とソース配布物には含まれません。
HY-Motion のコード、チェックポイント、出力には、地域および用途に関する制限を含む [Tencent HY-Motion 1.0 Community License](https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/License.txt) が適用されます。
HY-Motion をダウンロードまたは実行する前に、このライセンスを確認してください。

### ソースからの準備

HY-Motion のサブモジュールを初期化し、実行に必要な Git LFS アセットを取得してから、ロック済みの推論環境をインストールします。

```bash
./scripts/bootstrap_hy_motion.sh
uv sync --extra inference
```

ブートストラップスクリプトが取得するのは、実行時に使う統計ファイルとスケルトンアセットです。
モデルのチェックポイントは別にダウンロードします。

### モーション生成用チェックポイント

Lite またはフル版のチェックポイントをダウンロードします。
この README の例では、より小さい Lite モデルを使います。

```bash
uv run --extra inference huggingface-cli download \
  tencent/HY-Motion-1.0 \
  --include "HY-Motion-1.0-Lite/*" \
  --local-dir vendor/HY-Motion-1.0/ckpts/tencent
```

フル版を使う場合は、`HY-Motion-1.0-Lite/*` を `HY-Motion-1.0/*` に置き換えてください。
推論には大きなメモリが必要になるため、現在のハードウェアの目安は [HY-Motion のモデル一覧](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo) で確認してください。

Git のチェックアウトではなくローカルの wheel を使う場合は、`inference` を含めてインストールします。

```bash
python3.12 -m pip install \
  "/absolute/path/to/vroom-0.1.0-py3-none-any.whl[inference]"
```

HY-Motion は別の場所にクローンします。
続いて、[公式のセットアップ手順](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#get-started-with-hy-motion-10) に従って Git LFS アセットとモーション生成用チェックポイントを取得します。
そのうえで、HY-Motion のルートディレクトリとチェックポイントの場所を両方指定します。

```bash
export VROOM_HY_MOTION_ROOT=/absolute/path/to/HY-Motion-1.0
export HY_MOTION_MODEL_PATH=/absolute/path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite
```

コマンドごとに `--hy-motion-root` と `--model-path` を指定しても構いません。

### テキストエンコーダーのローカル配置

`--text-models` のデフォルト値 `auto` は、Qwen3-8B と CLIP-L のローカルディレクトリが両方そろっていれば、それらを使います。
どちらかがなければ、初回生成時に Hugging Face からモデルをダウンロードしてキャッシュします。

オフラインで使うために、両方のテキストエンコーダーをあらかじめダウンロードできます。

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

`--text-models local` を指定するとローカルのモデルディレクトリが必須になり、どちらかがなければエラーになります。
モデルの重みは Git の追跡対象外です。
`inference` には、大きな重みファイルの転送に使う Hugging Face の Xet ヘルパーも含まれます。

## アニメーションの生成

セットアップが終わったら、次のコマンドで VRMA ファイルを生成できます。

```bash
uv run --extra inference vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma
```

このコマンドは、デフォルトパスにある Lite チェックポイント、4 秒の動作時間、シード `42`、デバイスの自動選択を使います。
出力先のディレクトリがなければ、自動的に作成します。

デバイスの指定には、次の引数を使います。

| 対象 | 引数 | 動作 |
| --- | --- | --- |
| 自動選択 | `--device auto` | CUDA、MPS、CPU の順に利用可能なデバイスを選びます |
| Apple Silicon | `--device mps` | 拡散モデルを MPS、テキストエンコーダーを CPU で実行します |
| Apple Silicon で全処理を MPS | `--device mps --text-device same` | テキストエンコーダーも MPS での実行を試みます |
| NVIDIA GPU | `--device cuda:0` | 最初の CUDA デバイスを使います |
| CPU | `--device cpu` | 両方の処理を CPU で実行します |

`vroom` の MPS 対応は、CUDA または CPU だけを選択する HY-Motion の実装を補います。
具体的には、デバイス検出を補正し、MPS では CPU で乱数を生成します。
さらに、PyTorch の MPS CPU フォールバックを有効にして、Qwen テキストエンコーダーをデフォルトで CPU に置きます。
実際の生成速度は、使用する PyTorch と macOS のバージョンによって変わります。
`vroom` より先に Torch をインポートするアプリケーションへ組み込む場合は、プロセスの起動前に `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定してください。

PyPI から取得する固定バージョンの Torch は、Linux x86-64 で CUDA の依存パッケージを導入します。
リポジトリに含まれるロックは、Windows の CUDA 環境には対応していません。

`vroom` は、選択したモデル設定が対応する動作時間の範囲を検査します。
公開済みチェックポイントでは約 0.667 秒から 12 秒までを指定でき、出力は 30 fps です。
範囲外の値は丸めずにエラーとします。
HY-Motion は、プロンプトを 60 語未満の英語で書くよう推奨しています。
プロンプトには、外見、環境、カメラワークではなく、体幹と四肢の動きを記述します。
GPU メモリを抑える場合は、30 語未満のプロンプトと 5 秒未満の動作が公式の目安です。

## ファイルの確認と検証

```bash
uv run vroom inspect path/to/avatar.vrm
uv run vroom inspect output/wave.vrma --json
uv run vroom validate output/wave.vrma
uv run vroom validate path/to/motion-pack.zip
```

バリデーターは、GLB コンテナ、アクセサ、キーフレーム時刻を検査します。
クォータニオンが正規化されているか、ヒューマノイドボーンとノードの対応関係、`vroom` が実装するヒューマノイドアニメーションの制約も検査対象です。
とくに、ヒューマノイドボーンに `scale` トラックは使えず、`translation` トラックを持てるのは `hips` だけです。

従来の VRM 0.x アバターを読み込むときは、旧仕様の親指ボーン名を VRM 1.0 相当の名前へ置き換えます。

## 開発とテスト

短時間で完了するテストスイートは、Torch とモデルの重みを必要としません。

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest
```

モデルをダウンロードした後は、次のコマンドで推論スモークテストを実行できます。

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  uv run --extra inference pytest -m integration
```

エンドツーエンドの生成テストを特定のアクセラレーターで実行するには、`HY_MOTION_TEST_DEVICE=mps` または `HY_MOTION_TEST_DEVICE=cuda:0` を設定します。

`AliciaSolid_vrm-0.51.vrm` と `VRMA_MotionPack.zip` は、再配布に制限のあるローカルテストフィクスチャです。
そのため、Git の追跡対象から外し、配布パッケージにも含めていません。
ファイルがなければ、対応するテストはスキップします。

## HY-Motion から VRMA への変換

HY-Motion は、各回転を回転行列の 2 列を交互に並べた形式で保持します。
`vroom` は HY-Motion と同じ Gram-Schmidt 変換を適用し、glTF の `[x, y, z, w]` 順のクォータニオンへ変換したうえで、時間軸上の符号を連続させます。

HY-Motion の木製マネキンに含まれるスケルトンは、右手系、メートル単位、上方向が +Y、前方が +Z、解剖学上の左側が +X という VRM 1.0 の座標系をすでに使っています。
`vroom` が生成する身体動作に必要なのは回転とルートの平行移動だけなので、推論時にはメモリ消費の大きいメッシュスキニングを無効にします。
次に、順運動学によってレストポーズの床面オフセットを考慮しながら生成結果を接地し、フレーム間差分ではなく各キーフレームにおける `hips` のローカル平行移動値を書き出します。

出力は、公式の [VRM Animation 1.0 仕様](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0) と glTF 2.0 のアニメーション規則に従います。
