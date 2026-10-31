#!/bin/zsh
# Generate the seven v3 activity prompts with ARDY SOMA, two seeds each.
# Prompts, durations and the turn's heading constraint follow Kokage's
# assets/models/ardy_activity_clips_v3_manifest.json (feet-apart set); the
# listening and thinking entries are cuts of those takes.
# Usage: gen_soma.sh [seed ...]   (default seeds: 3 29)
# Environment: SOMA_PROBE_DIR (default: current directory) holds .venv/ and
# the ardy checkout and receives takes/; SOMA_PROBE_PYTHON and ARDY_REPO
# override the interpreter and the checkout. See README.md.
set -u
P=${SOMA_PROBE_DIR:-$PWD}
T=${0:A:h}
PY=${SOMA_PROBE_PYTHON:-$P/.venv/bin/python}
ARDY=${ARDY_REPO:-$P/ardy}
cd $ARDY || exit 1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} LOCAL_CACHE=true PYTORCH_ENABLE_MPS_FALLBACK=1
export ARDY_DEVICE=${ARDY_DEVICE:-cpu} ARDY_TEXT_ENCODER_DEVICE=${ARDY_TEXT_ENCODER_DEVICE:-mps}
seeds=("$@"); [[ ${#seeds} -eq 0 ]] && seeds=(3 29)
typeset -A text dur
text[idle]="A person stands relaxed in place, breathing calmly, arms loose at their sides."; dur[idle]=12
text[listening]="A person stands still facing someone, listening attentively and nodding slowly and repeatedly."; dur[listening]=12
text[thinking]="A person standing in place raises one hand to their chin and looks up, thinking hard."; dur[thinking]=10
text[speaking]="A person stands in place talking to someone in front of them, making small natural hand gestures."; dur[speaking]=12
text[settle]="A person standing in place lowers their head slightly with a small sigh, then looks up again."; dur[settle]=3
text[tapped]="A person standing in place is lightly tapped on the shoulder, looks toward it in mild surprise, then faces forward again."; dur[tapped]=2
text[turn]="A person standing in place turns around to face the opposite direction."; dur[turn]=2.5
mkdir -p $P/takes
for name in idle listening thinking speaking settle tapped turn; do
  for seed in $seeds; do
    out=$P/takes/$name-$seed
    [[ -f $out.npz ]] && { echo "$name-$seed exists"; continue }
    extra=()
    [[ $name == turn ]] && extra=(--constraints $T/turn_constraints.json)
    start=$(date +%s)
    $PY -I $T/generate_mps.py "${text[$name]}" --model soma --duration ${dur[$name]} \
      --seed $seed --output $out "${extra[@]}" > $out.log 2>&1
    echo "$name-$seed exit=$? $(( $(date +%s) - start ))s $(tail -1 $out.log)"
  done
done
echo ALL-DONE
