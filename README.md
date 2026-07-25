# vroom

**English** | [日本語](README.ja.md)

HY-Motion can turn text into human motion, but it does not produce a portable
VRM Animation file. `vroom` bridges that gap: it converts an English
description of human motion into an avatar-independent VRM Animation 1.0
(`.vrma`) file.

HY-Motion 1.0 generates the motion. `vroom` packages the result in a VRMA file
with a self-contained VRM humanoid T-pose hierarchy that a VRMA-aware player
can retarget to an avatar. It does not render or preview animations.

The GLB, VRM, and VRMA layer uses only the Python standard library. Torch and
the HY-Motion inference stack are isolated in the optional `inference` extra.
The generation process does not use the FBX SDK, Blender, a JavaScript
exporter, or a third-party Python VRM library.

## Features

- Read and write GLB 2.0 containers, including binary accessor decoding.
- Read humanoid metadata from VRM 0.x and VRM 1.0 files.
- Write VRM Animation 1.0 files and check supported metadata and
  binary-accessor constraints.
- Convert HY-Motion's 6-D rotations and map its SMPL-H bones to VRM bones.
- Select CUDA, MPS, or CPU automatically, or choose a device explicitly.
- Use a CLI with `generate`, `curate`, `inspect`, and `validate` commands.

Generated files contain a VRM humanoid T-pose hierarchy, `hips` translation, and
local quaternion tracks for the 22 body joints produced by the released
HY-Motion checkpoints. Finger bones are omitted because those checkpoints do
not generate finger rotations. The generator does not synthesize
facial-expression or look-at animation.

## Setup

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash, Git, and [Git LFS](https://git-lfs.com/)

Unless noted otherwise, the commands below assume a Git checkout and a Bash
shell. The `inspect` and `validate` commands need only the base package;
`generate` also needs the `inference` extra, the HY-Motion runtime, and model
weights.

### HY-Motion license

HY-Motion is an external runtime and is intentionally excluded from `vroom`
wheels and source distributions. Its code, checkpoints, and outputs are
subject to the
[Tencent HY-Motion 1.0 Community License](https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/License.txt),
which includes territorial and use restrictions. Review that license before
downloading or running HY-Motion.

### Source checkout

Initialize the HY-Motion submodule, fetch its required Git LFS runtime assets,
and install the locked inference environment:

```bash
./scripts/bootstrap_hy_motion.sh
uv sync --extra inference
```

The bootstrap script downloads the runtime statistics and skeleton assets, but
not the model checkpoint.

### Motion checkpoint

Download either the Lite or full checkpoint. The examples in this README use
the smaller Lite model:

```bash
uv run --extra inference huggingface-cli download \
  tencent/HY-Motion-1.0 \
  --include "HY-Motion-1.0-Lite/*" \
  --local-dir vendor/HY-Motion-1.0/ckpts/tencent
```

For the full model, replace `HY-Motion-1.0-Lite/*` with `HY-Motion-1.0/*`.
Inference is memory-intensive; consult the
[upstream model table](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo)
for current hardware guidance.

If you use a local wheel instead of a Git checkout, install it with the
`inference` extra:

```bash
python3.12 -m pip install \
  "/absolute/path/to/vroom-0.1.0-py3-none-any.whl[inference]"
```

Clone HY-Motion separately and follow its
[upstream setup instructions](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#get-started-with-hy-motion-10)
to fetch the Git LFS assets and download a motion checkpoint. Then configure
the HY-Motion root and checkpoint locations:

```bash
export VROOM_HY_MOTION_ROOT=/absolute/path/to/HY-Motion-1.0
export HY_MOTION_MODEL_PATH=/absolute/path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite
```

The same paths can be supplied per command with `--hy-motion-root` and
`--model-path`.

### Optional local text encoders

The default `--text-models auto` mode uses local Qwen3-8B and CLIP-L
directories when both are present. Otherwise, it downloads the models from
Hugging Face and caches them during the first generation.

To prepare both text encoders for offline use:

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

Pass `--text-models local` to require these local model directories and fail
if either is missing. Model weights are ignored by Git. The `inference` extra
includes Hugging Face's Xet transfer helper for large weight shards.

## Generate an animation

After setup, generate a VRMA file:

```bash
uv run --extra inference vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma
```

This uses the Lite checkpoint at its default path, a four-second duration,
seed `42`, and automatic device selection. The output directory is created
when needed.

Library callers that need several deterministic variants of the same prompt
can use `HyMotionBackend.generate_variants`. It accepts 1–16 unique seeds,
encodes a matching bounded prompt batch, and asks the already-loaded motion
pipeline for every seed in one pass. `generate` remains the single-seed
convenience API.

Useful device settings are:

| Target | Arguments | Behavior |
| --- | --- | --- |
| Automatic | `--device auto` | Prefers CUDA, then MPS, then CPU |
| Apple Silicon | `--device mps` | Runs diffusion on MPS and text encoding on CPU |
| Apple Silicon, all MPS | `--device mps --text-device same` | Tries the text encoder on MPS too |
| NVIDIA GPU | `--device cuda:0` | Uses the first CUDA device |
| CPU | `--device cpu` | Runs both stages on CPU |

`vroom`'s MPS support adapts upstream HY-Motion, which selects only CUDA or
CPU. It corrects upstream device discovery, uses CPU-seeded noise on MPS,
enables PyTorch's MPS CPU fallback, and keeps the Qwen text encoder on CPU by
default. Actual throughput varies with the installed PyTorch and macOS
versions. If an embedding application imports Torch before `vroom`, set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before starting the process.

The Torch version pinned from PyPI pulls in CUDA dependencies on Linux x86-64.
The committed lock does not provide a Windows CUDA environment.

`vroom` enforces the duration range defined by the selected model
configuration. The released checkpoints accept approximately 0.667 to
12 seconds and produce 30 fps; a value outside that range is rejected.
Upstream recommends English prompts shorter than 60 words and focused on body
and limb motion rather than appearance, environments, or camera directions.
Its lower-VRAM guidance uses prompts shorter than 30 words and motions shorter
than five seconds.

## Curate an animation

`curate` turns structurally valid, sampled LINEAR VRMA into a deterministic
rotation-only clip:

```bash
uv run vroom curate output/wave.vrma \
  --output output/wave-upper-body.vrma \
  --bones spine chest upperChest neck head leftUpperArm rightUpperArm \
  --duration 2.0 \
  --fade-in 0.25 \
  --fade-out 0.25 \
  --rotation-gain 1.25
```

The command drops root translation and all unselected tracks, expresses each
selected rotation as `inverse(first) * frame`, keeps quaternion signs on the
shortest path, and retimes the retained samples to the requested duration.
Smoothstep fades reach the exact identity rotation at both endpoints.
If an amplified or unfaded winding track cannot return to exact identity
without a quaternion sign discontinuity, curation fails and asks for a longer
fade-out or a narrower source crop.
`--rotation-gain` uses quaternion interpolation to scale first-frame-relative
rotations before those fades. It defaults to `1`, accepts values from `0`
(neutral) through `4`, and values above `1` deterministically amplify subtle
motion.
`--rotation-projection` accepts `none`, `twist-x`, `twist-y`, or `twist-z`.
The twist modes use swing-twist decomposition to retain only rotation around
the selected local bone axis before gain and fades. For example,
`--rotation-projection twist-y` makes head and neck tracks yaw-only without
Euler-angle conversion.
`--start-frame` and inclusive `--end-frame` can select a source range;
`--name` overrides the animation name. The output retains the humanoid nodes
needed for retargeting and records bounded source SHA-256, byte length, and
transform details under `asset.extras.vroomCuration`.

To keep local curation work bounded, the source limit is 64 MiB, 512 nodes,
18,000 samples per accessor, one million decoded numeric accessor components
in total, and 256 animation channels or samplers. Curated duration is limited
to 3,600 seconds, and the serialized output is limited to 16 MiB. Inputs
outside those limits fail before an output file is written.

## Inspect and validate files

```bash
uv run vroom inspect path/to/avatar.vrm
uv run vroom inspect output/wave.vrma --json
uv run vroom validate output/wave.vrma
uv run vroom validate path/to/motion-pack.zip
```

The validator checks the GLB container, accessors, keyframe times, quaternion
normalization, humanoid mappings, and the humanoid animation restrictions
implemented by `vroom`. In particular, humanoid bones may not have scale
tracks, and only `hips` may have a translation track.

When reading a legacy VRM 0.x avatar, `vroom` normalizes its legacy thumb names
to VRM 1.0 semantics.

## Development and tests

The fast test suite does not require Torch or model weights:

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest
```

After downloading the models, run the opt-in inference smoke test with:

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  uv run --extra inference pytest -m integration
```

Set `HY_MOTION_TEST_DEVICE=mps` or `HY_MOTION_TEST_DEVICE=cuda:0` to run the
end-to-end generation test on a specific accelerator.

`AliciaSolid_vrm-0.51.vrm` and `VRMA_MotionPack.zip` are optional local
fixtures with redistribution restrictions, so they are ignored by Git and
excluded from packages. Their tests skip when the files are absent.

## How conversion works

HY-Motion stores each rotation as two interleaved columns of a rotation
matrix. `vroom` applies the same Gram-Schmidt conversion as upstream, converts
the result to glTF's `[x, y, z, w]` quaternion order, and keeps quaternion
signs continuous over time.

The skeleton from HY-Motion's wooden mannequin already uses the VRM 1.0 basis:
right-handed, meters, +Y up, +Z forward, and anatomical left on +X. Because
the body motion generated by `vroom` needs only rotations and root
translation, it disables the memory-intensive mesh-skinning step during
inference. A forward-kinematics pass aligns the generated motion with the
floor while accounting for the source rest pose's floor offset, then writes
each frame's local `hips` translation rather than frame-to-frame deltas.

The output follows the official
[VRM Animation 1.0 specification](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)
and the glTF 2.0 animation rules.
