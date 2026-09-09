# ARDY verification — 2026-09-09

This record covers vroom's ARDY authoring and Core27-to-VRMA conversion.
It does not qualify every avatar, memory configuration, operating system,
or motion prompt. Generated motions and the restricted avatar fixture stay
in ignored local output directories.

## Runtime

The tested environment uses Python 3.12.11, ARDY 0.2.0 at
`693f74d13b3d04a0a22ce127ee79c929dd89756b`, Torch 2.8.0,
Transformers 5.8.1, PEFT 0.20.0, Accelerate 1.14.0, and Hugging Face Hub 1.30.0.
The committed `uv.lock` owns the complete dependency graph.
[The motion manifest](../src/vroom/ardy_models.json) and
[encoder loader](../src/vroom/ardy_loading.py) own the tested artifact pins.

Motion sampling uses float32. The local encoder uses float32 on CPU/MPS and
bfloat16 on CUDA, with the MNTP adapter merged and supervised adapter applied.
Native motion correction is enabled; export grounding is disabled.
All full-generation checks use the complete checkpoint denoising schedule.

## Automated checks

- Ruff and all 346 deterministic tests pass. The ordinary suite imports no
  inference runtime and performs no inference or downloads.
- Eighteen explicitly enabled CPU contracts pass against real ARDY: feature
  masks, FK, pose/joint/root conditioning, angular measurements, strict
  rejection, bidirectional Llama attention, and native continuation correction.
- The sdist builds into a wheel with both Core manifests and the authoring
  guide in the source distribution. Packages contain no weights, source
  motions, generated VRMA, or avatar fixtures.
- Existing HY-Motion options and deterministic behavior remain covered. Its
  resolved dependencies are unchanged; HY-Motion full inference was not rerun.

The full native test
[`test_ardy_integration.py`](../tests/test_ardy_integration.py) exercises:

1. Default ARDY selection, a three-second greeting, seed 42, VRMA validation,
   and a reusable source export with 61 samples and no prompt metadata.
2. Two five-second continuations, seeds 7 and 8, a prompt change at 3.5 seconds,
   and a full-body endpoint pose taken from the original source's first frame.
3. Exact preservation of all 61 prefix samples, 101-sample outputs, and finite
   constraint measurements. All 57 endpoint measurements per candidate pass
   the default 0.05 m / 0.1 rad tolerances on all four recorded model/device runs.

## Apple Silicon

Hardware: Apple M5 Max, `Mac17,7`, 128 GiB unified memory. OS: macOS 26.6.2.
Motion runs on MPS and text encoding on CPU. Both Core40 and Core8 pass the
full native test, including their corrected continuations.

These are individual runs with downloaded weights and warm filesystem caches.
Each timed backend call loads its models, encodes its prompts, samples motion,
corrects it, and constructs clips. Export and test overhead are excluded from
the call times. The second call produces both continuation candidates.

| Model | Three-second clip | Two continuations | Process peak RSS |
| --- | ---: | ---: | ---: |
| Core40 | 7.61 s | 6.82 s | 43.72 GiB |
| Core8 | 10.09 s | 8.81 s | 43.91 GiB |

An earlier cold Core40 call took 18.97 seconds. These measurements establish
neither sustained throughput nor a real-time guarantee. Peak RSS includes the
large CPU encoder; it is not GPU memory alone or the configurable motion-data
budget. Lower-memory Apple devices were not tested.

A separate strict Core40 plan walks to `[0, 0.7]` meters in XZ, stops, and
raises the right hand, with root/heading targets at 0, 2, and 4 seconds. All six
measurements pass: maximum position error 0.00706 m and heading error 0.02270 rad.
A 12-second Core40 walking clip also passes generation and VRMA validation
with 241 samples, exercising real history cropping beyond the ten-second model
window. That additional long clip was not separately visually reviewed.

Local receipts and outputs are under `output/ardy-verification/`:
`mps-core40-anchored/`, `mps-core8-anchored/`, and `mps-path-report.json`.
The local runs set `HF_HUB_CACHE` to this checkout's `output/huggingface`
directory and `HF_HUB_OFFLINE=1` after downloading the pinned weights.

## Linux / NVIDIA

Hardware: AMD Ryzen 7 9700X, 30.46 GiB OS-reported host memory, NVIDIA GeForce
RTX 5090 with 32 GiB VRAM. OS: Linux 7.0.0-28 x86-64, glibc 2.39; NVIDIA driver
580.173.02. Torch 2.8.0+cu128 includes the required `sm_120` architecture.
Both motion and text encoding run on CUDA device 0.

Both Core40 and Core8 pass the full native test. Each of the four continuation
candidates has 57 measurements within tolerance and an exact original prefix.
The remote deterministic suite passes 344 tests, with only the two optional
local avatar/motion-pack fixture tests skipped. All 18 CPU contracts pass too.

| Model | Three-second clip | Two continuations | Process peak RSS |
| --- | ---: | ---: | ---: |
| Core40 | 35.10 s | 32.23 s | 17.04 GiB |
| Core8 | 33.74 s | 42.26 s | 16.88 GiB |

Both runs peak at 14.07 GiB CUDA allocated memory and 14.28 GiB reserved memory.
These are single complete-backend call measurements, including loading and
encoding, under different host/cache conditions from the Mac. They are not an
isolated GPU-speed comparison. Each receipt records hashes of all 17 vroom
source files; they match the local implementation.

The remote environment, downloads, caches, temporary files, and outputs are
under `/home/xie/vroom`. Activate `.runtime/env.sh` there and set `PYTHONPATH=src`
when using `.venv-ardy/bin/python`. Copies of the receipts and generated clips
are under local `output/ardy-verification/cuda-core40/` and `cuda-core8/`.

## Rendered output

The temporary verification viewer uses Three.js 0.180.0 and
`@pixiv/three-vrm` / `@pixiv/three-vrm-animation` 3.4.2 with the existing local
`AliciaSolid_vrm-0.51.vrm` fixture. Synthetic conversion probes establish the
rest pose, thumb bases, left/right elbows, collapsed spine rotation, root
movement, and advancing playback in the actual VRMA player.

Core40's greeting reads as an anatomical right-hand wave; the tested Core8
sample is closer to a brief salute. Both models produce 24 rotation tracks
plus hips translation. Core40's initial pose has
its arm down; Core8's starts with its hand raised. Consequently, the explicit
endpoint pose in the continuation test returns Core8 to that raised-hand pose
even though the text asks for a lowered hand. Pose targets and text can conflict;
the endpoint test measures the requested source pose, not a guessed rest pose.

Correction receives every preserved prefix frame as a full-body constraint.
For the Core40 samples, the root step at 3.00–3.05 seconds is 0.70 mm for seed 7
and 1.29 mm for seed 8, with neighboring steps between 0.42 and 2.02 mm. Before
the prefix constraint fix those joins moved 11.0 and 41.1 mm. A dedicated native
regression also verifies the fix before exact-prefix restoration can hide it.
The two Core8 joins move 0.535 and 0.520 mm. In the rendered comparison, all four
corrected continuations remove the earlier visible body/foot pop. The largest
local joint changes at those joins are now upper-arm movement rather than the
earlier leg jump. The actual player loads all 25 tracks and advances playback
without page errors for all reviewed clips.
These samples do not establish seamless continuation for every generated motion.

CUDA-generated waves and seed 7 continuations for both models were also rendered
locally in the same player. All four loaded 25 tracks, advanced playback, and
showed no page error or visible join pop. The reviewed CUDA root steps are
2.251 mm for Core40 and 1.102 mm for Core8, consistent with their neighboring
steps. Core40 again gives the clearer right-hand wave; Core8's initial and
endpoint pose keeps the hand near the forehead. CUDA seed 8 continuations were
checked numerically but were not separately visually reviewed.

## Reproduction

Install the separate environment as described in the
[authoring guide](ardy.md). Obtain the encoder's gated model access first.
Run ordinary checks:

```bash
uv run ruff check src tests
uv run pytest
```

Run CPU contracts with `ARDY_ROOT` pointing to the parent of the installed
`ardy` package, or an upstream checkout at the pinned revision:

```bash
VROOM_TEST_ARDY_RUNTIME=1 ARDY_ROOT=/path/to/ardy/source \
  .venv-ardy/bin/python -m pytest tests/test_ardy_conditioning_integration.py
```

Run the full model check once per model/device combination:

```bash
VROOM_TEST_ARDY_MODELS=1 ARDY_TEST_DEVICE=mps ARDY_TEST_MODEL=core40 \
  ARDY_TEST_OUTPUT_DIR=output/ardy-verification/mps-core40 \
  .venv-ardy/bin/python -m pytest tests/test_ardy_integration.py
```

Use `core8` for the second model and `cuda:0` for the NVIDIA target. Set
`HF_HUB_OFFLINE=1` after all pinned artifacts are cached. Text placement defaults
to CPU for MPS and the selected CUDA device for CUDA. The same outputs can be
loaded into any compatible VRMA player for visual review; structural validation
alone does not establish motion quality or destination-avatar foot contacts.
