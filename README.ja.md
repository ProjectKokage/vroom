# vroom

[English](README.md) | **日本語**

`vroom` は、テキストから完成した人体動作クリップを生成します。
**デフォルトは ARDY** で、`--backend hy-motion` を指定すると HY-Motion 1.0 を使います。
`vroom` は、身体の回転とルートの平行移動を、アバターに依存しない [VRM Animation 1.0](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)（`.vrma`）ファイルへ書き出します。

VRMA 対応プレイヤーは、生成したクリップを VRM ヒューマノイドへ適用できます。
`vroom` が担うのは、アニメーションファイルの作成と検証です。
描画とプレビューは行いません。

## コマンド

| コマンド | 用途 |
| --- | --- |
| `generate` | 英語プロンプトまたは ARDY のモーション計画から VRMA クリップを生成します |
| `curate` | サンプリング済みの LINEAR 補間 VRMA から、選択した回転と任意の閉じたルートモーションを持つクリップを作ります |
| `inspect` | VRM または VRMA の GLB を要約します |
| `validate` | 一つ以上の VRMA ファイル、または ZIP 内の全 VRMA ファイルを検証します |

GLB、VRM、VRMA を扱うコードは Python 標準ライブラリだけを使い、実行時依存パッケージを持ちません。
基本パッケージだけで `curate`、`inspect`、`validate` を実行できます。
ARDY の推論環境はオプション依存グループ `ardy`、HY-Motion の推論環境は `inference` で導入します。
Torch とテキストエンコーダーの依存バージョンが異なるため、別々の環境にインストールしてください。
テキストのエンコードとモーションの推論は、どちらもローカルで実行します。

## 出力するデータ

| データ | 生成する VRMA の内容 |
| --- | --- |
| ヒューマノイド階層 | リターゲットに使う、自己完結した VRM の T ポーズノード |
| ルートモーション | `hips` のローカル平行移動 |
| 身体動作 | ARDY は Core27 から両親指の付け根を含む 24 個の VRM ボーントラックを出力。HY-Motion は身体の 22 関節を出力 |
| 指 | ARDY は親指の付け根のみ生成し、残りの指のトラックは省略。HY-Motion は指のトラックを生成しません |
| 表情と視線 | 表情アニメーションと LookAt アニメーションは生成しません |

## クイックスタート

### 必要なソフトウェア

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash、Git。HY-Motion のセットアップには [Git LFS](https://git-lfs.com/) も必要です
- ARDY のネイティブ補正処理をビルドする C++17 コンパイラーと CMake 3.15 以上

以下のコマンドは、Git からチェックアウトしたリポジトリと Bash を前提とします。

### ARDY のインストールと生成

ARDY は、アクセス承認が必要な [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) を基にした LLM2Vec テキストエンコーダーをローカルで使います。
Hugging Face のモデルページでアクセスを取得し、そのアカウントでログインしてください。

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv sync --extra ardy
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy hf auth login

UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

ARDY 環境には、固定したリビジョンの実行コード、スケルトンアセット、ネイティブ補正モジュールをインストールします。
初回生成時には、固定したリビジョンの Core チェックポイント、Llama 本体、LLM2Vec アダプターを Hugging Face から取得します。
8B テキストエンコーダーには大きなメモリとディスク容量が必要で、CPU に配置してもメモリ消費はなくなりません。
リモート推論サービスは使いません。
`vroom` はプロンプトを VRMA、任意で保存するソース NPZ、レポートへ書き込みません。

デフォルトは `--model core`（Core40）、4 秒、シード `42`、デバイスの自動選択です。
モーション補正は有効で、ソースのルートモーションを保持します。
Core40 と `--model core8` は、どちらも 20 fps で動作します。
`--ground` を指定すると、出力クリップ全体を垂直方向へ一度だけ移動して接地します。
`--no-postprocess` は ARDY の補正処理を無効にします。

ARDY のコードには [Apache 2.0](https://github.com/nv-tlabs/ardy/blob/main/LICENSE) が適用されます。
[Core のモデルカード](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40) は NVIDIA Open Model Agreement を参照しています。
Llama エンコーダーには別の利用条件とアクセス要件があるため、使う構成要素の公式条件を確認してください。
`vroom` の配布物にはモデルの重みを含めません。

### ARDY で動作を指定する

再利用できるソースモーションを保存し、一回のモデルロードで複数のシードを比較できます。

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person crouches, stands, and raises both arms." \
  --duration 3 --seeds 7 11 --device mps \
  --source-output output/raise-arms.npz \
  --output output/raise-arms.vrma
```

複数の候補には、`raise-arms-7.vrma`、`raise-arms-7.npz` のようにシードを付けた名前を使います。
`--plan motion.json` では、時刻ごとのプロンプト、ルートの通過点と向き、参照ポーズ、関節の位置と回転を指定できます。
`--continue-from source.npz` は、既存の ARDY ソースクリップを保持して続きを生成します。
`--report report.json` は制約からの誤差を保存し、`--strict` は許容誤差を超えた場合にファイルの出力前に失敗します。

JSON 形式、Python API、座標系、時刻、続きの生成、制約の限界は [ARDY オーサリングガイド](docs/ardy.md)（英語）を参照してください。

### ARDY のデバイス選択

| 引数 | 動作 |
| --- | --- |
| `--device auto` | CUDA、MPS、CPU の順に選びます |
| `--device mps` | Apple Silicon の MPS でモーション、CPU でテキストを処理します |
| `--device mps --text-device same` | 両方の処理に MPS を指定します |
| `--device cuda:0` | 指定した NVIDIA GPU で両方の処理を実行します |
| `--device cpu` | 両方の処理を CPU で実行します |

`--text-device` でエンコーダーの配置先を個別に指定できます。
エンコーダーは、モーション生成を始める前に解放します。
この表は実行経路を示すもので、性能を保証するものではありません。
モデル推論は対象マシンで別途検証する必要があります。
ARDY 環境は Torch 2.8.0 を使い、Linux x86-64 の wheel は CUDA 12.8 を含みます。
このロックファイルには Windows 環境を定義していません。

必要なモデルとエンコーダーを取得した後、キャッシュだけで読み込むには `HF_HUB_OFFLINE=1` を設定します。
独自のチェックポイント配置先とアセット検証については、[オーサリングガイド](docs/ardy.md)を参照してください。

## HY-Motion を使う

HY-Motion は、専用の `inference` 環境で `--backend hy-motion` を指定すると利用できます。
この節の引数と生成例は、HY-Motion 向けです。

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
uv run --extra inference vroom generate --backend hy-motion \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

`generate --backend hy-motion` は、デフォルトパスのフル版チェックポイント、4 秒の動作時間、シード `42`、デバイスの自動選択を使います。
出力先のディレクトリがなければ、コマンドが作成します。
HY-Motion のチェックアウト内に二つのテキストエンコーダーがそろっていない場合、初回生成時に Qwen3-8B と CLIP-L を Hugging Face からダウンロードし、Hugging Face のキャッシュへ保存します。

### HY-Motion のインストール先とモデルの配置

#### Lite 版のモーションチェックポイント

Lite 版もフル版と同じコマンドでダウンロードできます。
`HY-Motion-1.0/*` を `HY-Motion-1.0-Lite/*` に置き換えてください。
どちらのモデルも推論には大きなメモリが必要になるため、両チェックポイントのハードウェア要件は [HY-Motion のモデル一覧](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo) で確認してください。

#### wheel からのインストール

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

#### テキストエンコーダーのローカル配置

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

### HY-Motion のモーション生成

指定できる動作時間は、選択したチェックポイントが決めます。
公開済みチェックポイントは約 0.667 秒から 12 秒までを受け付け、30 fps のモーションを生成します。
チェックポイントの範囲から外れた動作時間は、`vroom` がエラーにします。

HY-Motion は、60 語未満の英語プロンプトを推奨しています。
外見、背景、カメラワークを避け、体幹と四肢の動きを記述してください。
メモリ消費を抑える場合、HY-Motion は一つのシード、30 語未満のプロンプト、5 秒未満の動作を推奨しています。

出力を再現したい場合やデバイスを固定したい場合は、引数を指定します。

```bash
uv run --extra inference vroom generate --backend hy-motion \
  "A person crouches, stands, and raises both arms." \
  --duration 3 \
  --seed 7 \
  --device mps \
  --output output/raise-arms.vrma
```

CFG スケール、ODE 推論ステップ数、テキストエンコーダーの配置、モデルパス、接地処理を含む全引数は、`uv run --extra inference vroom generate --backend hy-motion --help` で確認できます。

一つのプロンプトから、シードの異なる再現可能な結果を複数作る場合、Python から `HyMotionBackend.generate_variants` を呼び出せます。
このメソッドは、重複しない 1 個から 16 個のシードを受け取り、最大 16 件からなるプロンプトのバッチを一度エンコードして、一回のモデルロード後に全シードを生成します。
単一シード用の API は `generate` です。

#### HY-Motion のデバイス選択

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

`curate` は、構造を検証できるサンプリング済みの LINEAR 補間 VRMA を受け取り、選択した回転トラックを持つ決定論的なクリップを書き出します。
デフォルトでは従来どおり回転専用となり、ルートの平行移動を削除します。

```bash
uv run vroom curate output/wave.vrma \
  --output output/wave-upper-body.vrma \
  --bones spine chest upperChest neck head leftUpperArm rightUpperArm \
  --duration 2.0 \
  --fade-in 0.25 \
  --fade-out 0.25 \
  --rotation-gain 1.25
```

デフォルトのコマンドは、ルートの平行移動と指定外のトラックを削除します。
選択した回転を `inverse(first) * frame` で表し、クォータニオンの符号を最短経路上で連続させたうえで、残したサンプルの時間を変更します。
Smoothstep フェードにより、最初と最後のキーフレームは恒等回転になります。

増幅したトラックやフェードしていないトラックでは、巻き込みを含む回転から恒等回転へ戻る連続なクォータニオン経路を作れない場合があります。
その場合、コマンドは出力を書き出さずに失敗し、より長いフェードアウトまたは狭いソース範囲を求めます。

| 引数 | 効果 |
| --- | --- |
| `--bones` | 指定したヒューマノイド回転トラックを残します |
| `--start-frame`、`--end-frame` | 終了フレームを含むソース範囲を選びます |
| `--duration` | サンプル間の相対時間を保ったままメインのサンプル時間を変更します。復帰テールは別途時間を追加します |
| `--fade-in`、`--fade-out` | 始点と終点に Smoothstep フェードを適用します |
| `--rotation-gain` | 先頭フレームを基準にした回転を `0` から `4` の範囲で拡大または縮小し、デフォルト値には `1` を使います |
| `--rotation-projection` | `none` は回転全体を保ち、`twist-x`、`twist-y`、`twist-z` はローカル軸まわりのねじりだけを残します |
| `--bone-rotation-gain BONE=GAIN` | 選択した 1 ボーンのゲインを上書きします。ボーンごとに繰り返せます |
| `--bone-rotation-projection BONE=PROJECTION` | 選択した 1 ボーンの射影を上書きします。ボーンごとに繰り返せます |
| `--hips-translation` | デフォルトの `strip` またはオプトインの `relative-return` を選びます |
| `--hips-return-duration` | `relative-return` に必須の復帰テール時間を設定します |
| `--name` | 出力アニメーション名を設定します |

回転ゲインは、フェードより前にクォータニオン補間を使って適用します。
`1` より大きい値は、小さな動きを増幅します。
ねじりへの射影には Swing-Twist 分解を使うため、`--rotation-projection twist-y` を指定すると、オイラー角へ変換せずに頭と首のトラックをヨー回転だけにできます。

ボーン単位のオプションは、指定した選択済みボーンだけについてグローバルなゲインまたは射影を上書きします。
これにより、脚、足先、腰、腕を含む全身のボーンを残しながら、頭だけを特定軸へ射影できます。

```bash
uv run vroom curate output/bow.vrma \
  --output output/bow-full-body.vrma \
  --bones hips spine chest upperChest neck head \
          leftUpperLeg rightUpperLeg leftLowerLeg rightLowerLeg \
          leftFoot rightFoot leftToes rightToes \
          leftUpperArm rightUpperArm leftLowerArm rightLowerArm \
  --duration 2.0 \
  --fade-in 0 \
  --fade-out 0 \
  --bone-rotation-projection head=twist-x \
  --bone-rotation-gain leftUpperLeg=0.9 \
  --bone-rotation-gain rightUpperLeg=0.9 \
  --hips-translation relative-return \
  --hips-return-duration 0.5
```

`relative-return` は、切り出した各 `hips` サンプルを `rest + (frame - first)` として残します。
メインのルート軌道をフェードしたり短縮したりしません。
最初の LINEAR 区間がレスト位置から動く場合は、その区間と同じ長さのレスト保持サンプルを先頭へ追加します。
メインの最終サンプル後は、進入速度を引き継ぐ三次 Hermite 曲線をサンプリングし、速度 0 でレスト位置へ戻します。
ルートの最後の 2 サンプルは正確なレスト位置になり、選択したすべての回転にも滑らかな Hermite 復帰を追加して、最後の 2 サンプルを恒等回転にします。
したがって出力時間は、メインの `--duration`、必要な先頭保持時間、`--hips-return-duration` の合計です。

復帰テールはメインのサンプルを弱めたり削除したりしません。
ルートの復帰は、メイン軌道に対して 25 パーセントまたは 5 cm の大きい方まで外側へ継続できますが、キュレーション後のルート軌道全体はレスト位置から 0.5 m 以内でなければなりません。
この上限を超える動作は拒否します。
回転の復帰はメインの最大値を超えてもかまいませんが、有限で、恒等回転へ向かう連続した最短クォータニオン経路上になければなりません。
Kokage などの利用側は、測定済みの出力メタデータを使って、アセットごとにより厳しい上限を適用できます。
従来のフェードではなく復帰テールで回転を恒等姿勢へ戻す場合は、`--fade-out 0` を指定します。

出力には、リターゲットに必要なヒューマノイドノードを残します。
ソースの SHA-256、バイト長、変換設定は `asset.extras.vroomCuration` に記録します。
メタデータのバージョン 3 は、選択した各ボーンの実効ポリシーに加え、ルートのモード、基準、保持時間、復帰時間、サンプル数、補間方法、終端条件を記録します。
トップレベルでは `perBonePolicies`、`hipsTranslationPolicy`、`mainDurationMicroseconds`、`returnDurationMicroseconds`、`durationMicroseconds`、`maximumRootDisplacementMeters`、`maximumRootVelocityMetersPerSecond` という閉じたフィールド名を使います。
`durationMicroseconds` は先頭の保持時間を含む最終エンコード出力から測定します。
ルートの最大変位と区間速度も最終 float32 出力から測定し、float32 で値を下回らない境界へ切り上げます。
削除モードでは `hipsTranslationPolicy` を `"none"` とし、復帰時間と二つのルート境界を `0` にします。
Hermite 復帰の詳細は `transform.hipsTranslationReturn` に残します。

### Python キュレーション API

Python からも同じ閉じたポリシー型を使います。
ボーン単位で上書きしない選択済みボーンには、グローバルな回転設定を適用します。

```python
from vroom import BoneCurationPolicy, HipsTranslationPolicy, curate_vrma

curated = curate_vrma(
    "output/bow.vrma",
    bones=(
        "hips",
        "spine",
        "leftUpperLeg",
        "rightUpperLeg",
        "leftFoot",
        "rightFoot",
    ),
    duration=2.0,
    fade_out=0.0,
    bone_policies=(
        BoneCurationPolicy("leftUpperLeg", rotation_gain=0.9),
        BoneCurationPolicy("rightUpperLeg", rotation_gain=0.9),
    ),
    hips_translation_policy=HipsTranslationPolicy(
        mode="relative-return",
        return_duration=0.5,
    ),
)
```

`BoneCurationPolicy` のボーン名は重複できず、`bones` に含まれていなければなりません。
`HipsTranslationPolicy()` のデフォルトは `strip` なので、既存の API 呼び出しは平行移動を削除し、従来の終端フェード動作を維持します。

### キュレーションの上限

`curate` は、次の上限を超えた入力を、出力の書き込み前に拒否します。

| 対象 | 上限 |
| --- | ---: |
| ソースファイル | 64 MiB |
| ノード | 512 |
| アクセサごとのサンプル | 18,000 |
| デコードする数値アクセサ成分の合計 | 1,000,000 |
| アニメーションのチャンネルまたはサンプラー | 256 |
| キュレーション後のサンプル | 18,000 |
| キュレーション後の長さ | 3,600 秒 |
| `relative-return` のルート変位 | レスト位置から 0.5 m |
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

ARDY の CPU 契約テストは、実際のランタイムと Core スケルトン、合成した統計値を使います。
モデルの重みと推論は使いません。
`ARDY_ROOT` に、固定したリビジョンの公式コードのチェックアウトを指定してください。

```bash
VROOM_TEST_ARDY_RUNTIME=1 ARDY_ROOT=/absolute/path/to/ardy \
  UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy \
  pytest tests/test_ardy_conditioning_integration.py
```

HY-Motion のモデルをダウンロードした後は、オプトイン形式の推論スモークテストを実行できます。

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
  uv run --extra inference pytest -m integration
```

エンドツーエンドの生成テストでアクセラレーターを指定するには、`HY_MOTION_TEST_DEVICE=mps` または `HY_MOTION_TEST_DEVICE=cuda:0` を設定します。

`AliciaSolid_vrm-0.51.vrm` と `VRMA_MotionPack.zip` は、再配布に制限のあるローカルテストフィクスチャです。
Git は両ファイルを追跡せず、配布パッケージにも含めません。
ファイルがなければ、対応するテストをスキップします。
