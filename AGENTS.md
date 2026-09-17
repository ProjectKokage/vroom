# Working in vroom

vroom is a Python CLI/library for generating, curating, inspecting and validating
VRM Animation files. ARDY is the default backend; HY-Motion remains an explicit
alternative. vroom does not render or preview motion.

## Start here

Read [README](README.md) for commands, output contracts and current limits.
For ARDY changes, read the relevant [authoring guide](docs/ardy.md) sections and
[verification record](docs/ardy-verification.md). Inspect the pinned upstream
implementation when changing backend behavior. Preserve unrelated edits and
unpublished commits; state the affected behavior and checks before editing.

## Essential contracts

- Keep GLB/VRM/VRMA and base CLI operations usable without Torch, models or
  network inference. Optional backends stay behind their existing interfaces.
- Use Python and dependency versions from [pyproject.toml](pyproject.toml) and
  [uv.lock](uv.lock). Keep ARDY and HY-Motion inference environments separate;
  their Torch/text-encoder dependencies differ. Do not upgrade pins casually.
- Preserve avatar-independent VRMA, humanoid mappings, coordinate conventions,
  quaternion order/continuity and rest-pose retargeting. Only hips may carry
  humanoid translation; humanoid scale tracks are invalid.
- Treat files, archives, arrays and motion plans as untrusted. Preserve size,
  expansion, accessor and output bounds, finite-value validation and failure
  before output publication. Do not weaken validation to admit a fixture.
- Keep deterministic seed/curation behavior and provenance metadata. Preserve
  curation's endpoint, root-trajectory and return-tail contracts in README.
  Do not store prompts in generated VRMA, source NPZ or reports.
- Keep models, caches, native build output and restricted avatar/motion fixtures
  out of Git and distributions. Respect backend/model notices and access terms;
  source availability does not imply model or output redistribution permission.
- Real inference can download large or gated artifacts. Use only the explicit
  setup/generation scope authorized by the user; ordinary validation must not
  trigger downloads. Cached-only runs use the documented offline settings.

## Verification

From the repository root, resolve the committed dev environment when needed
with `uv sync --locked --group dev`. For Python code changes, run focused tests,
then:

```sh
uv run --locked ruff check src tests
uv run --locked pytest
```

Ordinary tests need no model weights or Torch. Follow README for opt-in ARDY
runtime and backend inference checks when those boundaries change. State the
exact runtime/model/device exercised and report missing fixtures as skips,
not successful inference. Format validation does not establish rendered motion
quality; consumer/rendered checks have their own evidence scope.

For user-visible CLI changes, update both README languages and relevant command
help. For prose-only guides or agent instructions, check links, command paths
and `git diff --check`; no inference environment is needed.

## Delivery

Branch names must not begin with `codex` (case-insensitive), including
`codex/` and `codex-`. Rename tool-generated defaults before committing or
pushing; use a descriptive name such as `docs-agent-guides`.

Use a task branch, keep diffs focused and review them before handoff. Commit
only task files; push, merge or publish only when authorized. Continue routine
implementation within scope and get approval before changing product or asset
policy. Report the change, checks and remaining evidence gaps.
