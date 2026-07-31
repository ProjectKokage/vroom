# vroom

**English** | [日本語](README.ja.md)

HY-Motion 1.0 generates human motion from text. Its output requires humanoid
bone mappings and a portable file before a VRM application can
retarget it. `vroom` writes the body rotations and root translation as an
avatar-independent
[VRM Animation 1.0](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)
(`.vrma`) file.

A VRMA-aware player can apply the clip to a VRM humanoid. `vroom` creates and
checks animation files. It does not render or preview them.

## Commands

| Command | Purpose |
| --- | --- |
| `generate` | Generate one VRMA clip from an English motion prompt |
| `curate` | Turn sampled LINEAR VRMA into a bounded clip with selected rotation tracks and no translation |
| `inspect` | Summarize a VRM or VRMA GLB |
| `validate` | Validate one or more VRMA files, or every VRMA file in a ZIP archive |

The GLB, VRM, and VRMA code uses the Python standard library and no runtime
dependencies. The base package runs `curate`, `inspect`, and `validate`. The
`inference` extra adds Torch and the HY-Motion stack for `generate`. The
generation path stays inside the Python process, where `vroom` writes the GLB
data.

## Output scope

| Data | Generated VRMA |
| --- | --- |
| Humanoid hierarchy | Self-contained VRM T-pose nodes for retargeting |
| Root motion | Local `hips` translation |
| Body motion | Local quaternion tracks for the 22 joints produced by the released HY-Motion checkpoints |
| Fingers | Omitted because the released checkpoints do not generate finger rotations |
| Face and gaze | No expression or look-at animation |

## Quick start

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash, Git, and [Git LFS](https://git-lfs.com/)

The commands below assume a Git checkout and a Bash shell.

### Read the HY-Motion license

`vroom` does not bundle HY-Motion code or checkpoints in its wheels and source
distributions. HY-Motion code, checkpoints, and outputs fall under the
[Tencent HY-Motion 1.0 Community License](https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/License.txt),
which contains territorial and use restrictions. Read that license before you
download or run HY-Motion.

### Install and generate

Initialize the HY-Motion submodule, fetch its Git LFS runtime assets, and
install the locked inference environment:

```bash
./scripts/bootstrap_hy_motion.sh
uv sync --extra inference
```

The bootstrap script fetches the runtime statistics and skeleton assets. It
does not fetch a motion checkpoint. Download the full checkpoint:

```bash
uv run --extra inference huggingface-cli download \
  tencent/HY-Motion-1.0 \
  --include "HY-Motion-1.0/*" \
  --local-dir vendor/HY-Motion-1.0/ckpts/tencent
```

Generate and validate a clip:

```bash
uv run --extra inference vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

`generate` uses the full checkpoint at its default path, a four-second
duration, seed `42`, and device auto-selection. The command creates the output
directory when needed. Unless both text encoders exist in the HY-Motion
checkout, the first generation downloads Qwen3-8B and CLIP-L from
Hugging Face and places them in the Hugging Face cache.

## Installation and model paths

### Lite motion checkpoint

The Lite model uses the same download command as the full model. Replace
`HY-Motion-1.0/*` with `HY-Motion-1.0-Lite/*`. Inference with either model
needs substantial memory; the
[upstream model table](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo)
lists hardware guidance for both checkpoints.

### Wheel installation

A wheel contains `vroom`, but not the HY-Motion runtime. Install the wheel with
the `inference` extra:

```bash
python3.12 -m pip install \
  "/absolute/path/to/vroom-0.1.0-py3-none-any.whl[inference]"
```

Clone HY-Motion in a separate directory and follow its
[setup instructions](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#get-started-with-hy-motion-10)
to fetch the Git LFS assets and a motion checkpoint. Then set both paths:

```bash
export VROOM_HY_MOTION_ROOT=/absolute/path/to/HY-Motion-1.0
export HY_MOTION_MODEL_PATH=/absolute/path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0
```

You can pass the same locations to one command with `--hy-motion-root` and
`--model-path`.

### Local text encoders

`--text-models auto` uses local Qwen3-8B and CLIP-L directories when both
exist. If either directory is missing, HY-Motion downloads the encoders from
Hugging Face during the first generation.

Download both encoders into the locations that `auto` detects:

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

Use `--text-models local` to require those directories and fail before
inference if either one is missing. Git ignores downloaded model weights. The
`inference` extra includes Hugging Face's Xet transfer helper for large weight
shards.

## Generate a motion

The selected checkpoint controls the accepted duration range. The released
checkpoints accept about 0.667 to 12 seconds and produce 30 fps motion.
`vroom` rejects a duration outside the checkpoint's range.

HY-Motion recommends an English prompt shorter than 60 words. Describe the
torso and limb movement rather than appearance, scenery, or camera movement.
Upstream's lower-memory guidance uses one seed, a prompt shorter than 30
words, and a motion shorter than five seconds.

Use explicit options when you need repeatable output or a fixed device:

```bash
uv run --extra inference vroom generate \
  "A person crouches, stands, and raises both arms." \
  --duration 3 \
  --seed 7 \
  --device mps \
  --output output/raise-arms.vrma
```

Run `uv run --extra inference vroom generate --help` for the complete option
list, including CFG scale, ODE inference steps, text-encoder placement, model
paths, and ground alignment.

Python callers that need several deterministic variants of one prompt can use
`HyMotionBackend.generate_variants`. It accepts 1 to 16 unique seeds, encodes
a prompt batch with at most 16 entries, and generates every seed after one
model load. `generate` provides the single-seed API.

### Device selection

| Target | Arguments | Behavior |
| --- | --- | --- |
| Auto | `--device auto` | Chooses CUDA, then MPS, then CPU |
| Apple Silicon | `--device mps` | Runs diffusion on MPS and text encoding on CPU |
| Apple Silicon, all MPS | `--device mps --text-device same` | Attempts text encoding on MPS |
| NVIDIA GPU | `--device cuda:0` | Uses the first CUDA device |
| CPU | `--device cpu` | Runs diffusion and text encoding on CPU |

HY-Motion's upstream device helper selects CUDA or CPU. `vroom` adds MPS
device discovery, creates MPS noise from a CPU seed, enables PyTorch's MPS CPU
fallback, and places the Qwen encoder on CPU by default. Throughput depends on
the installed PyTorch and macOS versions.

An embedding application may import Torch before `vroom` can enable the MPS
fallback. In that case, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before starting
the process.

The pinned PyPI Torch build pulls CUDA dependencies on Linux x86-64. The
committed lock does not define a Windows CUDA environment.

## Curate a motion

`curate` accepts sampled LINEAR VRMA that passes the structural checks and
writes a deterministic clip with selected rotation tracks and no translation:

```bash
uv run vroom curate output/wave.vrma \
  --output output/wave-upper-body.vrma \
  --bones spine chest upperChest neck head leftUpperArm rightUpperArm \
  --duration 2.0 \
  --fade-in 0.25 \
  --fade-out 0.25 \
  --rotation-gain 1.25
```

The command removes root translation and unselected tracks. It expresses each
selected rotation as `inverse(first) * frame`, keeps quaternion signs on the
shortest path, and retimes the retained samples. Smoothstep fades place the
first and last keyframes at the identity rotation.

An amplified or unfaded winding track may lack a continuous quaternion path
back to exact identity. The command then fails before writing output and asks
for a longer fade-out or a narrower source crop.

| Option | Effect |
| --- | --- |
| `--bones` | Keeps the listed humanoid rotation tracks |
| `--start-frame`, `--end-frame` | Selects a source range; the end frame is inclusive |
| `--duration` | Retimes the selected samples while preserving their relative spacing |
| `--fade-in`, `--fade-out` | Applies smoothstep fades at the endpoints |
| `--rotation-gain` | Scales first-frame-relative rotation from `0` through `4`; the default is `1` |
| `--rotation-projection` | Keeps all rotation with `none`, or keeps local-axis twist with `twist-x`, `twist-y`, or `twist-z` |
| `--name` | Sets the output animation name |

Rotation gain uses quaternion interpolation before the fades. A value above
`1` amplifies subtle motion. Twist projection uses swing-twist decomposition,
so `--rotation-projection twist-y` can keep yaw and remove the other head and
neck rotation without an Euler-angle conversion.

The output retains the humanoid nodes needed for retargeting. It records the
source SHA-256, source byte length, and transform settings under
`asset.extras.vroomCuration`.

### Curation limits

`curate` rejects oversized input before it writes output:

| Resource | Limit |
| --- | ---: |
| Source file | 64 MiB |
| Nodes | 512 |
| Samples per accessor | 18,000 |
| Decoded numeric accessor components | 1,000,000 |
| Animation channels or samplers | 256 |
| Curated duration | 3,600 seconds |
| Serialized output | 16 MiB |

## Inspect and validate files

```bash
uv run vroom inspect path/to/avatar.vrm
uv run vroom inspect output/wave.vrma --json
uv run vroom validate output/wave.vrma
uv run vroom validate path/to/motion-pack.zip
```

The validator checks the GLB container, accessors, keyframe times, quaternion
normalization, humanoid mappings, and the humanoid animation restrictions
implemented by `vroom`. Humanoid bones cannot carry scale tracks. `hips` is
the sole humanoid bone that can carry a translation track.

When `vroom` reads a VRM 0.x avatar, it maps the legacy thumb bone names to VRM
1.0 semantics.

## How HY-Motion becomes VRMA

HY-Motion stores each rotation as two interleaved columns of a rotation
matrix. `vroom` applies HY-Motion's Gram-Schmidt conversion, converts the
result to glTF's `[x, y, z, w]` quaternion order, and preserves quaternion
sign continuity over time.

HY-Motion's wooden mannequin skeleton uses the VRM 1.0 basis:
right-handed, meters, +Y up, +Z forward, and anatomical left on +X. VRMA body
motion needs rotations and root translation, so `vroom` disables the
memory-intensive mesh-skinning step during inference. A forward-kinematics
pass accounts for the source rest pose's floor offset, places the motion on
the floor, and writes each keyframe's local `hips` translation.

## Development

The fast test suite needs no Torch installation or model weights:

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest
```

After you download the models, run the opt-in inference smoke tests:

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
  uv run --extra inference pytest -m integration
```

Set `HY_MOTION_TEST_DEVICE=mps` or `HY_MOTION_TEST_DEVICE=cuda:0` to select an
accelerator for the end-to-end generation test.

`AliciaSolid_vrm-0.51.vrm` and `VRMA_MotionPack.zip` are optional local
fixtures with redistribution restrictions. Git ignores them, packages exclude
them, and the related tests skip when the files are absent.
