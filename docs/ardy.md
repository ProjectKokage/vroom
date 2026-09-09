# ARDY authoring

`vroom generate` uses ARDY to produce a finished VRMA clip. The same backend
can follow a timed plan, generate seed variants, or extend a saved source
motion. See the [README](../README.md#install-ardy-and-generate) for installation
and Hugging Face access setup.
The [verification record](ardy-verification.md) describes tested native and
rendered behavior for specific models and machines.

## Runtime and checkpoints

Keep ARDY in its own environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv sync --extra ardy
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate --help
```

The `ardy` and HY-Motion `inference` extras have incompatible upstream
dependencies. Switching extras in one environment changes the installed
stack. Use `.venv-ardy` for ARDY and `.venv` for HY-Motion or base tooling.
ARDY builds a native motion-correction extension, requiring a C++17 compiler
and CMake >= 3.15. macOS needs the Xcode command-line tools; Linux needs its
compiler toolchain. The pinned Python dependency is in
[`pyproject.toml`](../pyproject.toml), with resolved dependencies in `uv.lock`.

| `--model` | Released model | Rate | Generation horizon |
| --- | --- | --- | --- |
| `core` or `core40` | `ARDY-Core-RP-20FPS-Horizon40` | 20 fps | 40 frames / 2 seconds |
| `core8` | `ARDY-Core-RP-20FPS-Horizon8` | 20 fps | 8 frames / 0.4 seconds |

The default is Core40. A horizon is the model's generation chunk, not a total
clip-duration limit. `vroom` carries motion history through bounded model
windows to produce the requested duration. Both models use Core27. Robot G1
models do not have the human skeleton needed by this VRMA conversion.

First use downloads the selected pinned motion model into the Hugging Face
cache. [`ardy_models.json`](../src/vroom/ardy_models.json) owns checkpoint
revisions, filenames, byte lengths, and SHA-256 values. `vroom` checks every
listed motion artifact before loading it. `--checkpoints-dir /path/to/models`
instead reads a directory containing the released model folder, for example
`/path/to/models/ARDY-Core-RP-20FPS-Horizon40/`. Those local files must match the
same manifest.

The local text encoder loads pinned revisions of Meta Llama 3 8B Instruct and
the McGill LLM2Vec MNTP and supervised adapters. Revision identities live in
[`ardy_loading.py`](../src/vroom/ardy_loading.py). The base model requires
Hugging Face access and login. `HF_HUB_OFFLINE=1` requests cached-only loading;
all selected motion and encoder artifacts must already be cached. ARDY does
not use HY-Motion's `--text-models`, `--model-path`, or `--hy-motion-root` options.

Text encoding and motion inference stay local. The encoder is released before
motion sampling. `vroom` writes only the requested outputs: it does not add
prompts to VRMA, source exports, or diagnostic reports. A JSON plan is a file
the caller explicitly authors and may itself contain prompts.

## One prompt, variants, and continuation

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person walks forward and comes to a stop." \
  --duration 4 --seed 42 \
  --output output/walk.vrma \
  --source-output output/walk.npz
```

`--seeds 7 11 19` requests up to 16 unique seeds. One model load and one set of
prompt embeddings serve all candidates. With multiple seeds, both VRMA and
source filenames gain `-SEED` before their extensions. A single candidate
uses the given filename unchanged. Seeds select reproducible variants within
the chosen runtime; different devices or dependency versions can change
numeric results.

Preserve the generated source when further authoring is useful. A VRMA has
already collapsed the source hierarchy and cannot serve as an ARDY reference.
Extend the original motion using its `.npz` file:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  "A person stands still and raises the right hand." \
  --continue-from output/walk.npz --duration 6 \
  --output output/walk-and-wave.vrma \
  --source-output output/walk-and-wave.npz
```

`--duration` is the total duration, including the preserved prefix. The source
must use the model's 20 fps, contain at least four frames, and be shorter than
the requested total. Correction receives the original prefix as full-body
constraints, and every original source rotation and root sample is restored
afterward. This preserves the prefix exactly; it does not
guarantee a smooth boundary with the newly generated frames. Inspect the seam.

## JSON motion plan

Save this as `motion.json`:

```json
{
  "duration": 4,
  "prompts": [
    {"time": 0, "text": "A person walks forward."},
    {"time": 2, "text": "A person stops and raises the right hand."}
  ],
  "seeds": [7, 11],
  "constraints": [
    {"type": "root", "time": 0, "position": [0, 0], "heading": 0},
    {"type": "root", "time": 2, "position": [0, 0.7], "heading": 0},
    {"type": "root", "time": 4, "position": [0, 0.7], "heading": 0}
  ],
  "strict": false
}
```

```bash
UV_PROJECT_ENVIRONMENT=.venv-ardy uv run --extra ardy vroom generate \
  --plan motion.json --output output/planned.vrma \
  --source-output output/planned.npz --report output/planned-report.json
```

`duration` and `prompts` are required. `seeds` defaults to `[42]`,
`constraints` to `[]`, and `strict` to `false`. Optional `continuation` names
an NPZ prefix. Source paths inside the plan resolve relative to the plan's
directory. With `--plan`, put prompt, duration, seeds, and continuation in the
file; passing the corresponding CLI arguments is an error. `--strict` can
strengthen a loaded plan without editing the file.

All times are seconds on the model's frame grid: multiples of 0.05 seconds
at 20 fps. A duration includes both its first and last sample, so four seconds
has 81 frames. The first prompt starts at zero, later cues increase strictly,
and no cue or target exceeds the duration. With several prompts, the first
must last at least one four-frame token (0.2 seconds). Prompts longer than the
encoder's document/context limits fail instead of being silently shortened.

Plans reject unknown keys, duplicate JSON keys, malformed numbers, duplicate
seeds, and conflicting targets. The plan limit is 1 MiB and 4,096 authoring
items. `--motion-memory-mib` defaults to 256 MiB and bounds estimated working
motion data and source archive expansion. Increase it explicitly for longer
clips. It excludes model weights, encoder memory, and accelerator attention
allocations; it is not a limit on the process's total memory.
Output and report paths must be distinct from each other and from plan/source
inputs, including symbolic links and hard links to the same file.

## Constraint fields and coordinates

Positions use the ARDY Core source space: meters, right-handed, +Y up, +Z
forward, and anatomical left along +X. Root targets contain `[x, z]`; joint
and pose roots use `[x, y, z]`. Heading is radians around Y, with zero facing
+Z. Joint rotations are normalized **world-space** `[x, y, z, w]` quaternions.
They are not local bone rotations.

| Type | Required fields beyond `type` and `time` | Optional fields |
| --- | --- | --- |
| `root` | `position: [x, z]` | `heading` |
| `pose` | `source: "clip.npz"`, `sourceTime` | `joints: ["leftHand", ...]` |
| `joint` | `joint`, `rootPosition: [x, y, z]`, `rootHeading`, and at least one of `position` or `rotation` | The other of `position` or `rotation` |

Every type also accepts `positionTolerance` in meters and `rotationTolerance`
in radians. Defaults are 0.05 m and 0.1 rad. Tolerances govern diagnostics and
strict failure; they do not change the strength of the conditioning.

Joint names accept exact Core names, such as `LeftHand`, or the retained VRM
semantics, such as `leftHand`. `chest` resolves to `Spine2`, `upperChest` to
`Spine3`, and `spine` to `Spine`. `Spine1` remains addressable by its Core name.
Unsupported finger names fail; Core27 has only one joint per thumb and no
other finger joints.

A `pose` target uses forward kinematics on one source sample to constrain
positions and rotations. Without `joints`, it constrains all 27 Core joints.
A joint subset still constrains the source root's position and heading,
because ARDY requires root context for world-space targets. `sourceTime` must
fall on the reference file's own frame grid and within its samples.

For example, these targets reuse a saved pose and specify one wrist target:

```json
[
  {"type": "pose", "time": 0, "source": "output/walk.npz", "sourceTime": 0},
  {
    "type": "joint", "time": 2, "joint": "rightHand",
    "rootPosition": [0, 1, 0.7], "rootHeading": 0,
    "position": [-0.4, 1.5, 0.9],
    "rotation": [0, 0, 0, 1],
    "positionTolerance": 0.08
  }
]
```

The array belongs in a plan's `constraints` field. Positions in this example
illustrate the schema; choose reachable targets for the source character.
Root waypoints constrain their stated times and do not define an exact path
between them. Matching pose targets at both endpoints can request a return
to the same pose. There is no built-in equality constraint for endpoint
velocity, no guaranteed seamless loop, and no automatic rest-pose insertion.

## Correction, measurements, and export

ARDY correction runs by default. `--no-postprocess` disables it.
`--cfg-scale` controls text guidance and `--constraint-cfg-scale` controls
constraint guidance; both default to 2. `--steps` must be within the selected
checkpoint's denoising schedule; the default uses its full schedule.

Root, full-body pose, and named hand/foot pose targets also feed the upstream
correction constraints. Arbitrary joint targets condition generation but
have no corresponding generic correction lock. Correction can therefore
move them. In all cases, `vroom` measures the final corrected source motion
and reports deviations. `--strict` raises an error before writing any
candidate outputs if a measured error exceeds its tolerance. Without it,
generation writes the clips and prints missed targets. `--report` optionally
writes all measurements for every seed; it contains no prompt text.

Measurements use Core source coordinates **before optional export grounding
and avatar retargeting**. They do not prove that a differently proportioned
VRM hand reaches the same world position. Compare `error` with `tolerance` in
the JSON report; `joint` identifies the Core bone for joint measurements.
Position errors are distances, rotation errors are angular
distances, and heading differences wrap at ±π.

Export preserves the source root translation by default. `--ground` shifts
the entire exported clip vertically so its lowest source joint touches the
floor. It leaves the reusable source NPZ and constraint measurements in the
original source coordinates. It does not perform destination-avatar IK.

Core27 becomes 24 VRM rotation tracks. `vroom` removes two hand end markers
and folds `Spine1` into `Spine2` by composing rotations. The removed spine
pivot cannot retain its exact independent position, but its rotation is
preserved. Source bind bases, including nonidentity thumb bases, normalize to
VRM rest orientation. Quaternion signs remain continuous across frames.
Faces, gaze, and non-thumb fingers have no generated tracks.

## Source NPZ and Python API

`--source-output` writes an explicit reusable source export. It stores
float32 `local_rot_mats` with shape `[frames, 27, 3, 3]`, `root_positions` with
shape `[frames, 3]`, scalar `fps`, `skeleton="core27"`, `source_version=1`, and
the model identifier. It contains no prompt. The reader also accepts upstream
Core NPZ files with the required arrays and rate. It rejects pickle/object
arrays, non-finite data, invalid rotation matrices, unknown fields, and
archives that exceed the configured expanded-data budget.

Use typed plans from Python when embedding generation:

```python
from pathlib import Path

from vroom.ardy import ArdyBackend, ArdyOptions
from vroom.ardy_plan import ArdyPlan, PromptCue, RootTarget
from vroom.ardy_source import write_source
from vroom.vrma import write_vrma

plan = ArdyPlan(
    duration=4,
    prompts=(PromptCue(0, "A person walks forward and stops."),),
    seeds=(42,),
    constraints=(RootTarget(4, (0, 0.7), heading=0),),
)
backend = ArdyBackend(ArdyOptions(device="mps"))
for result in backend.generate_plan(plan):
    write_vrma(Path("output") / f"walk-{result.seed}.vrma", result.clip, result.skeleton)
    write_source(Path("output") / f"walk-{result.seed}.npz", result.source, model="core")
    for deviation in result.deviations:
        print(deviation.component, deviation.error, deviation.satisfied)
```

`generate_plan` returns one `ArdyGeneration` per seed with the VRM `clip`,
`skeleton`, reusable `source`, measured `deviations`, and `seed`. Python callers
control file writes. `generate(prompt, duration=..., seed=...)` returns one
`MotionClip`; `generate_variants(prompt, duration=..., seeds=...)` returns a
tuple of clips. `load_plan(path)` in `vroom.ardy_plan` reads the same JSON
schema used by the CLI. If an embedding application imports Torch first, set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before starting the process when MPS CPU fallback
is required.
