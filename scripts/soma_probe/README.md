# ARDY SOMA probe

Scripts used on October 31, 2026 to measure what the NVIDIA ARDY SOMA
checkpoints (released October 15, 2026) add over the Core40 clips that Kokage
ships. The result is recorded in Kokage's
`docs/evidence/2026-10-31-ardy-soma-probe.md`: the SOMA model predicts 30
joints without finger chains and pastes one relaxed-hand rest pose onto the
77 output joints, so fingers do not move, and the resting posture is the
Core40 stance. These scripts reproduce that measurement. They are not part
of the `vroom` package and do not write VRMA.

## Files

| File | Purpose |
| --- | --- |
| `gen_soma.sh` | Generates the seven v3 activity prompts at their v3 durations for seeds 3 and 29 into `takes/<name>-<seed>.npz` with a log each |
| `generate_mps.py` | Upstream `scripts/generate.py` from `nv-tlabs/ardy` `e706e27` (Apache-2.0, header retained) with two changes marked `Kokage probe`: the device is read from `ARDY_DEVICE`, and the text encoder is built on `ARDY_TEXT_ENCODER_DEVICE` |
| `turn_constraints.json` | The v3 turn's heading constraint: frames 0 and 74 of the 75-frame take, heading 0 to 180 degrees, root fixed |
| `measure.py` | Per take: finger rotation range, arm angle from down, elbow bend, thigh outward angle, knee/ankle/hip-joint separation, head yaw and pitch range; writes `measurements.json` beside the takes |
| `vrm_map.py` | Static SOMA 77-joint to VRM 1.0 humanoid bone map (55 bones, 9 folded, 13 end markers dropped); writes `vrm_map.json` into the current directory |
| `check_skeleton.py` | Checks the joint-name table against generated arrays: bone lengths, parent order, and finger rotations constant over frames and across takes |

## Environment

The SOMA checkpoints need `ardy` at `e706e27` or later. The `ardy` extra of
this repository pins `693f74d`, which has no SOMA model, so the probe used
its own environment:

- Python 3.12 venv with `torch==2.8.0` and `transformers` 5.8.1, and a
  sparse checkout of `nv-tlabs/ardy` at `e706e27` installed with
  `uv pip install -e ./ardy` (the G1 robot meshes are not needed).
- `nvidia/ARDY-SOMA-RP-30FPS-Horizon60` at revision `2442150`, 769 MB, and
  the text encoder used for the v3 clips: LLM2Vec on
  `meta-llama/Meta-Llama-3-8B-Instruct` revision `8afb486c` with the
  McGill-NLP `mntp` and `mntp-supervised` adapters. Both are gated or large;
  download them deliberately, then run offline (`gen_soma.sh` sets
  `HF_HUB_OFFLINE=1` unless overridden).
- The motion model runs on the CPU because its skeleton keeps float64
  buffers, which MPS refuses; the text encoder runs on MPS in float32. On an
  Apple M5 Max a take needs about 20 s after the one-time encoder load.

## Run

```bash
export SOMA_PROBE_DIR=/path/to/probe   # holds .venv/, ardy/ and receives takes/
zsh scripts/soma_probe/gen_soma.sh      # seeds 3 and 29; pass seeds to change
cd "$SOMA_PROBE_DIR"
python -I /path/to/vroom/scripts/soma_probe/measure.py \
  idle-3=takes/idle-3.npz idle-29=takes/idle-29.npz   # one label=path per take
python -I /path/to/vroom/scripts/soma_probe/check_skeleton.py takes/*.npz
python -I /path/to/vroom/scripts/soma_probe/vrm_map.py
```

`measure.py` and `check_skeleton.py` need only numpy. Generated `.npz`
files are ignored by this repository.
