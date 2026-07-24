# vroom

`vroom` turns an English motion prompt into a portable VRM Animation 1.0
(`.vrma`) file. HY-Motion 1.0 generates the motion; this project supplies a
dependency-free Python GLB/VRM/VRMA implementation and the retargetable
humanoid skeleton.

The format core uses only the Python standard library. Torch and the
HY-Motion inference stack are isolated in the optional `inference` extra.
There is no FBX SDK, Blender process, JavaScript exporter, or third-party
Python VRM library in the generation path.

## What is implemented

- GLB 2.0 container reading/writing and binary accessor decoding.
- VRM 0.x and VRM 1.0 humanoid metadata reading.
- VRM Animation 1.0 writing and binary-aware validation.
- Exact HY-Motion 6-D rotation conversion and SMPL-H-to-VRM bone mapping.
- CUDA, MPS, and CPU device selection. `auto` prefers CUDA, then MPS, then CPU.
- A `generate`, `inspect`, and `validate` command-line interface.
- UV-locked Python 3.12 environments and optional model dependencies.

Generated files contain a canonical VRM 1.0 T-pose hierarchy, hips translation,
and local quaternion tracks for the 22 body joints produced by HY-Motion.
Finger bones are omitted because HY-Motion 1.0 does not generate finger
rotations. Expressions and LookAt animation are not synthesized.

## Setup

HY-Motion is an external runtime and is intentionally not bundled in `vroom`
wheels or source distributions. Its code, checkpoints, and outputs are subject
to the upstream Community License, including territorial and use restrictions.
Review that license before downloading or running HY-Motion.

From a Git checkout of this repository, initialize the submodule and its small
Git LFS runtime assets:

```bash
./scripts/bootstrap_hy_motion.sh
```

If you installed `vroom` from a wheel, clone HY-Motion separately and fetch the
same Git LFS runtime assets following its upstream setup instructions. Then
point `vroom` at that checkout either for the process:

```bash
export VROOM_HY_MOTION_ROOT=/absolute/path/to/HY-Motion-1.0
```

or for one generation command:

```bash
vroom generate "A person jumps." \
  --hy-motion-root /absolute/path/to/HY-Motion-1.0 \
  --model-path /absolute/path/to/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  --output output/jump.vrma
```

Install the locked inference environment:

```bash
uv sync --extra inference
```

Download either the Lite or full motion checkpoint. Lite is the practical
default, especially on Apple Silicon:

```bash
uv run --extra inference huggingface-cli download \
  tencent/HY-Motion-1.0 \
  --include "HY-Motion-1.0-Lite/*" \
  --local-dir vendor/HY-Motion-1.0/ckpts/tencent
```

For the full model, replace `HY-Motion-1.0-Lite/*` with
`HY-Motion-1.0/*`.

By default, Qwen3-8B and CLIP-L are loaded from Hugging Face and cached on
first generation. To make those models explicitly local:

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

Then pass `--text-models local`. Model weights are intentionally ignored by
Git. The inference extra includes Hugging Face's Xet transfer helper for the
large weight shards. Review the
[HY-Motion repository](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) for
its license and hardware guidance.

## Generate an animation

```bash
uv run --extra inference vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --model-path vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  --duration 4 \
  --seed 42 \
  --device auto \
  --output output/wave.vrma
```

Useful device choices:

```bash
# Apple Silicon: diffusion on MPS, text encoding on CPU (the safe default)
uv run --extra inference vroom generate "A person jumps." \
  --model-path vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  --device mps --output output/jump.vrma

# Try putting the text encoder on MPS as well
uv run --extra inference vroom generate "A person jumps." \
  --model-path vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  --device mps --text-device same --output output/jump.vrma

# NVIDIA
uv run --extra inference vroom generate "A person jumps." \
  --model-path vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  --device cuda:0 --output output/jump.vrma
```

MPS support is a compatibility layer around upstream HY-Motion, which only
selects CUDA or CPU. `vroom` corrects upstream device discovery, uses
CPU-seeded noise on MPS, enables PyTorch MPS CPU fallback, and keeps the
Qwen text encoder on CPU by default. Real model throughput still depends on
the particular PyTorch/macOS combination. When embedding `vroom` in a process
that imports Torch first, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before startup.

The locked PyPI Torch build supplies CUDA dependencies on Linux x86-64.
Windows CUDA requires selecting PyTorch's CUDA package index and regenerating
the UV lock for that platform.

HY-Motion clamps its learned sequence length. The released checkpoints accept
approximately 0.667–12 seconds and emit 30 fps. Prompts work best in English,
under 30 words, and should describe body motion rather than appearance,
objects, or camera work.

## Inspect and validate files

```bash
uv run vroom inspect AliciaSolid_vrm-0.51.vrm
uv run vroom inspect output/wave.vrma --json
uv run vroom validate output/wave.vrma
uv run vroom validate VRMA_MotionPack.zip
```

The validator checks the GLB container, accessors, keyframe times, quaternion
normalization, humanoid mappings, and VRMA's animation restrictions. In
particular, humanoid bones may not have scale tracks and only hips may have a
translation track.

The supplied Alicia avatar is a legacy VRM 0.x file. The reader normalizes its
legacy thumb names to VRM 1.0 semantics. Generated animations stay
avatar-independent; a VRMA-aware player performs the final retarget.

## Development and tests

The fast suite does not install Torch:

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest
```

To run the opt-in real inference smoke test after downloading models:

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite \
  uv run --extra inference pytest -m integration
```

Set `HY_MOTION_TEST_DEVICE=mps` or `HY_MOTION_TEST_DEVICE=cuda:0` to run that
end-to-end generation smoke test on a specific accelerator.

`AliciaSolid_vrm-0.51.vrm` and `VRMA_MotionPack.zip` are local,
redistribution-restricted fixtures and are ignored by Git. Fixture tests skip
when they are absent. The motion-pack test reads each VRMA member in memory and
does not extract or reproduce it.

## Design notes

HY-Motion stores each rotation as two interleaved columns of a rotation
matrix. `vroom` applies the same Gram–Schmidt conversion as upstream, converts
the result to glTF's `[x, y, z, w]` quaternion order, and fixes quaternion sign
continuity over time.

The vendored wooden skeleton already uses the VRM 1.0 basis: right-handed,
meters, +Y up, +Z forward, and anatomical left on +X. The costly wooden-mesh
skinning step is disabled during inference. A small forward-kinematics pass
grounds the generated joints while accounting for the source mesh's rest-floor
offset, then writes absolute hips translations as required by glTF animation.

The output follows the official
[VRM Animation 1.0 specification](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)
and glTF 2.0 animation rules.
