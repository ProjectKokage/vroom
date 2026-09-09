# vroom

**English** | [日本語](README.ja.md)

`vroom` generates finished human-motion clips with **ARDY by default**, or
HY-Motion 1.0 with `--backend hy-motion`. It writes body rotations and root
translation as an avatar-independent
[VRM Animation 1.0](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm_animation-1.0)
(`.vrma`) file.

A VRMA-aware player can apply the clip to a VRM humanoid. `vroom` creates and
checks animation files. It does not render or preview them.

## Commands

| Command | Purpose |
| --- | --- |
| `generate` | Generate VRMA clips from an English prompt or an ARDY motion plan |
| `curate` | Turn sampled LINEAR VRMA into a bounded clip with selected rotations and optional closed root motion |
| `inspect` | Summarize a VRM or VRMA GLB |
| `validate` | Validate one or more VRMA files, or every VRMA file in a ZIP archive |

The GLB, VRM, and VRMA code uses the Python standard library and no runtime
dependencies. The base package runs `curate`, `inspect`, and `validate`. The
`ardy` extra installs the ARDY stack; `inference` retains the HY-Motion stack.
Install them in separate environments because their Torch and text-encoder
dependencies differ. Text encoding and motion inference run locally.

## Output scope

| Data | Generated VRMA |
| --- | --- |
| Humanoid hierarchy | Self-contained VRM T-pose nodes for retargeting |
| Root motion | Local `hips` translation |
| Body motion | ARDY: 24 VRM bone tracks from Core27, including both thumb roots; HY-Motion: 22 body joints |
| Fingers | ARDY animates thumb roots only; remaining fingers have no generated tracks. HY-Motion has no finger tracks |
| Face and gaze | No expression or look-at animation |

## Quick start

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Bash and Git; [Git LFS](https://git-lfs.com/) for HY-Motion setup
- A C++17 compiler and CMake >= 3.15 for ARDY's native motion-correction build

The commands below assume a Git checkout and a Bash shell.

### Install ARDY and generate

ARDY uses a local LLM2Vec text encoder based on the gated
[Meta-Llama-3-8B-Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
Obtain access on its Hugging Face page, then log in using that account:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv sync --extra ardy
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy hf auth login

UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

The ARDY environment installs the pinned upstream runtime, including its
skeleton assets and native correction extension. First generation downloads
the pinned Core checkpoint, Llama base model, and LLM2Vec adapters from Hugging
Face. The 8B text encoder requires substantial memory and disk space; placing
it on CPU does not remove its memory cost. No remote inference service is used.
`vroom` does not store prompts in VRMA, optional source NPZ files, or reports.

Defaults are `--model core` (Core40), four seconds, seed `42`, automatic device
selection, motion correction enabled, and source root motion preserved. Both
Core40 and `--model core8` run at 20 fps. `--ground` applies a single vertical
shift to the exported clip; `--no-postprocess` disables ARDY correction.

ARDY code uses [Apache 2.0](https://github.com/nv-tlabs/ardy/blob/main/LICENSE).
The [Core model card](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40)
links its NVIDIA Open Model Agreement. The Llama encoder has separate terms
and access requirements; read the linked upstream terms for the components
you use. `vroom` packages contain no model weights.

### Author with ARDY

Save reusable source motion and compare several seeds after one model load:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person crouches, stands, and raises both arms." \
  --duration 3 --seeds 7 11 --device mps \
  --source-output output/raise-arms.npz \
  --output output/raise-arms.vrma
```

Multiple candidates receive seed suffixes, such as `raise-arms-7.vrma` and
`raise-arms-7.npz`. Use `--plan motion.json` for timed prompt changes, root
waypoints and headings, reference poses, and joint position/rotation targets.
`--continue-from source.npz` preserves and extends a prior ARDY source clip.
`--report report.json` saves measured target errors; `--strict` fails before
writing outputs if any target misses tolerance.

The [ARDY authoring guide](docs/ardy.md) covers the JSON schema, Python API,
coordinate conventions, timing, continuation, and constraint limitations.

### ARDY device selection

| Arguments | Behavior |
| --- | --- |
| `--device auto` | Chooses CUDA, then MPS, then CPU |
| `--device mps` | Motion on Apple Silicon MPS; text encoding on CPU |
| `--device mps --text-device same` | Requests both stages on MPS |
| `--device cuda:0` | Motion and text encoding on the selected NVIDIA GPU |
| `--device cpu` | Both stages on CPU |

`--text-device` can explicitly place the encoder on another supported device.
The encoder is released before motion generation. These are execution paths,
not performance guarantees; model inference needs separate verification on
the target machine. The ARDY extra pins Torch 2.8.0, whose Linux x86-64 wheel
includes CUDA 12.8. No Windows environment is defined by this lock.

For cached-only model loading, set `HF_HUB_OFFLINE=1` after the selected model
and encoder artifacts have been downloaded. Custom checkpoint directories
and artifact verification are documented in the [authoring guide](docs/ardy.md).

## Use HY-Motion

HY-Motion remains available through `--backend hy-motion` in its own
`inference` environment. All options and examples in this section select HY-Motion.

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
uv run --extra inference vroom generate --backend hy-motion \
  "A person takes three steps forward, waves with the right hand, and stops." \
  --output output/wave.vrma

uv run vroom validate output/wave.vrma
```

`generate --backend hy-motion` uses the full checkpoint at its default path, a four-second
duration, seed `42`, and device auto-selection. The command creates the output
directory when needed. Unless both text encoders exist in the HY-Motion
checkout, the first generation downloads Qwen3-8B and CLIP-L from
Hugging Face and places them in the Hugging Face cache.

### HY-Motion installation and model paths

#### Lite motion checkpoint

The Lite model uses the same download command as the full model. Replace
`HY-Motion-1.0/*` with `HY-Motion-1.0-Lite/*`. Inference with either model
needs substantial memory; the
[upstream model table](https://github.com/Tencent-Hunyuan/HY-Motion-1.0#-model-zoo)
lists hardware guidance for both checkpoints.

#### Wheel installation

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

#### Local text encoders

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

### Generate a HY-Motion clip

The selected checkpoint controls the accepted duration range. The released
checkpoints accept about 0.667 to 12 seconds and produce 30 fps motion.
`vroom` rejects a duration outside the checkpoint's range.

HY-Motion recommends an English prompt shorter than 60 words. Describe the
torso and limb movement rather than appearance, scenery, or camera movement.
Upstream's lower-memory guidance uses one seed, a prompt shorter than 30
words, and a motion shorter than five seconds.

Use explicit options when you need repeatable output or a fixed device:

```bash
uv run --extra inference vroom generate --backend hy-motion \
  "A person crouches, stands, and raises both arms." \
  --duration 3 \
  --seed 7 \
  --device mps \
  --output output/raise-arms.vrma
```

Run `uv run --extra inference vroom generate --backend hy-motion --help` for the complete option
list, including CFG scale, ODE inference steps, text-encoder placement, model
paths, and ground alignment.

Python callers that need several deterministic variants of one prompt can use
`HyMotionBackend.generate_variants`. It accepts 1 to 16 unique seeds, encodes
a prompt batch with at most 16 entries, and generates every seed after one
model load. `generate` provides the single-seed API.

#### HY-Motion device selection

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
writes a deterministic clip with selected rotation tracks. By default it keeps
the original rotation-only behavior and removes root translation:

```bash
uv run vroom curate output/wave.vrma \
  --output output/wave-upper-body.vrma \
  --bones spine chest upperChest neck head leftUpperArm rightUpperArm \
  --duration 2.0 \
  --fade-in 0.25 \
  --fade-out 0.25 \
  --rotation-gain 1.25
```

The default command removes root translation and unselected tracks. It
expresses each selected rotation as `inverse(first) * frame`, keeps quaternion
signs on the shortest path, and retimes the retained samples. Smoothstep fades
place the first and last keyframes at the identity rotation.

An amplified or unfaded winding track may lack a continuous quaternion path
back to exact identity. The command then fails before writing output and asks
for a longer fade-out or a narrower source crop.

| Option | Effect |
| --- | --- |
| `--bones` | Keeps the listed humanoid rotation tracks |
| `--start-frame`, `--end-frame` | Selects a source range; the end frame is inclusive |
| `--duration` | Retimes the selected main samples while preserving their relative spacing; a return tail adds time |
| `--fade-in`, `--fade-out` | Applies smoothstep fades at the endpoints |
| `--rotation-gain` | Scales first-frame-relative rotation from `0` through `4`; the default is `1` |
| `--rotation-projection` | Keeps all rotation with `none`, or keeps local-axis twist with `twist-x`, `twist-y`, or `twist-z` |
| `--bone-rotation-gain BONE=GAIN` | Overrides gain for one selected bone; repeat for different bones |
| `--bone-rotation-projection BONE=PROJECTION` | Overrides projection for one selected bone; repeat for different bones |
| `--hips-translation` | Uses the default `strip` mode or opt-in `relative-return` mode |
| `--hips-return-duration` | Sets the return-tail duration required by `relative-return` |
| `--name` | Sets the output animation name |

Rotation gain uses quaternion interpolation before the fades. A value above
`1` amplifies subtle motion. Twist projection uses swing-twist decomposition,
so `--rotation-projection twist-y` can keep yaw and remove the other head and
neck rotation without an Euler-angle conversion.

Per-bone options override the global gain or projection only for their named
selected bone. This lets one curation retain a full-body bone set while, for
example, projecting head motion without projecting the hips, legs, feet, or
arms:

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

`relative-return` retains each cropped hips sample as
`rest + (frame - first)`. It does not fade or shorten the main root trajectory.
If the first LINEAR segment leaves rest, curation prepends one rest hold with
the same duration as that segment. After the final main sample, it continues
the incoming velocity and samples a cubic Hermite return to rest with zero
terminal velocity. The last two root samples are exact rest, and every
selected rotation receives a corresponding smooth Hermite return whose last
two samples are exact identity. The output duration is therefore the requested
main `--duration`, plus the optional start hold, plus
`--hips-return-duration`.

The return tail never attenuates or removes a main sample. A root return may
continue beyond the main trajectory by at most the larger of 25 percent or
5 cm, and the entire curated root trajectory must remain within 0.5 m of rest.
Curation rejects motion beyond those caps. Rotation returns may pass the main
peak but must remain finite and on a continuous shortest quaternion path to
identity. Kokage or another consumer can enforce tighter per-asset bounds from
the measured output metadata. Set `--fade-out 0` when the return tail, rather
than the legacy main-clip fade, should bring rotations back to identity.

The output retains the humanoid nodes needed for retargeting. It records the
source SHA-256, source byte length, and transform settings under
`asset.extras.vroomCuration`. Metadata version 3 records the effective policy
for every selected bone and the root mode, basis, hold, return duration, sample
count, interpolation, and endpoint contract.

The version 3 top-level policy and bound fields have this closed vocabulary:

```json
{
  "perBonePolicies": [
    {
      "bone": "leftUpperLeg",
      "rotationGain": 0.9,
      "rotationProjection": "none"
    }
  ],
  "hipsTranslationPolicy": "relative-return",
  "mainDurationMicroseconds": 2000000,
  "returnDurationMicroseconds": 500000,
  "durationMicroseconds": 2533333,
  "maximumRootDisplacementMeters": 0.03999999910593033,
  "maximumRootVelocityMetersPerSecond": 0.11374999582767487
}
```

`durationMicroseconds` is measured from the final encoded output and includes
the optional start hold. Root displacement and segment velocity maxima are
also measured from the final float32 output and rounded upward to a float32
bound. Strip mode writes `hipsTranslationPolicy: "none"`, a zero return
duration, and zero root bounds. Hermite construction details remain under
`transform.hipsTranslationReturn`.

### Python curation API

Python callers use the same closed policies. Global rotation settings remain
the fallback for selected bones that have no override:

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

`BoneCurationPolicy` names must be unique members of `bones`.
`HipsTranslationPolicy()` defaults to `strip`, so existing API calls keep
removing translation and using the original endpoint-fade behavior.

### Curation limits

`curate` rejects oversized input before it writes output:

| Resource | Limit |
| --- | ---: |
| Source file | 64 MiB |
| Nodes | 512 |
| Samples per accessor | 18,000 |
| Decoded numeric accessor components | 1,000,000 |
| Animation channels or samplers | 256 |
| Curated samples | 18,000 |
| Curated duration | 3,600 seconds |
| `relative-return` root displacement | 0.5 m from rest |
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

The opt-in ARDY CPU contract checks use the real runtime and Core skeleton,
with synthetic statistics and no model weights or inference. Point `ARDY_ROOT`
at the pinned upstream checkout:

```bash
VROOM_TEST_ARDY_RUNTIME=1 ARDY_ROOT=/absolute/path/to/ardy \
  UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy \
  pytest tests/test_ardy_conditioning_integration.py
```

After you download the HY-Motion models, run its opt-in inference smoke tests:

```bash
HY_MOTION_MODEL_PATH=vendor/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
  uv run --extra inference pytest -m integration
```

Set `HY_MOTION_TEST_DEVICE=mps` or `HY_MOTION_TEST_DEVICE=cuda:0` to select an
accelerator for the end-to-end generation test.

`AliciaSolid_vrm-0.51.vrm` and `VRMA_MotionPack.zip` are optional local
fixtures with redistribution restrictions. Git ignores them, packages exclude
them, and the related tests skip when the files are absent.
