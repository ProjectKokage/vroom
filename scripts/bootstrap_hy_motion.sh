#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HY_ROOT="$PROJECT_ROOT/vendor/HY-Motion-1.0"

# Avoid downloading all LFS objects during submodule checkout.
GIT_LFS_SKIP_SMUDGE=1 \
  git -C "$PROJECT_ROOT" submodule update --init --recursive

# This is local to the vendored repository.
git -C "$HY_ROOT" lfs install --local

# Retrieve the runtime statistics and wooden-body assets.
# Pulling all dump_wooden .bin files is simpler and less fragile than
# maintaining an exact per-file list.
git -C "$HY_ROOT" lfs pull \
  --include="stats/*.npy,scripts/gradio/static/assets/dump_wooden/*.bin" \
  --exclude=""

# Fail early if LFS left pointer files behind.
HY_ROOT="$HY_ROOT" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["HY_ROOT"])

required = [
    root / "stats/Mean.npy",
    root / "stats/Std.npy",
    root / "scripts/gradio/static/assets/dump_wooden/v_template.bin",
    root / "scripts/gradio/static/assets/dump_wooden/j_template.bin",
    root / "scripts/gradio/static/assets/dump_wooden/skinWeights.bin",
    root / "scripts/gradio/static/assets/dump_wooden/skinIndice.bin",
    root / "scripts/gradio/static/assets/dump_wooden/kintree.bin",
    root / "scripts/gradio/static/assets/dump_wooden/faces.bin",
]

for path in required:
    if not path.is_file():
        raise SystemExit(f"Missing HY-Motion asset: {path}")

    prefix = path.read_bytes()[:80]
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise SystemExit(f"Unresolved Git LFS pointer: {path}")

print("HY-Motion LFS assets are present.")
PY
