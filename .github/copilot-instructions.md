# TeraiNet Project Guidelines

TeraiNet is a lightweight experimental project for wildlife species classification from
camera-trap imagery. Keep changes easy to understand and review.

## Architecture

- Put reusable Python logic in `src/terainet/`; keep notebooks as experiment and
  orchestration layers.
- Treat `config.yaml` as the source of dataset paths, class definitions, and run settings.
- Preserve label ordering and preprocessing contracts across data preparation, training,
  evaluation, and inference.

## Change Scope

- Make the smallest change that correctly fulfills the request and preserves existing behavior.
- Reuse existing patterns before adding abstractions or dependencies.
- If a broader refactor would materially improve correctness, testability, or maintainability,
  explain its evidence and propose it as separate follow-up work; do not silently expand scope.
- Do not modify unrelated files.

## ML Safety

- Keep training, validation, test, and out-of-distribution evaluation data distinct.
- Prevent leakage, including location-level spatial leakage when location metadata is available.
- Keep validation and inference deterministic; apply augmentation only to training data.
- Make inputs, random seeds, label mappings, and generated artifacts explicit and reproducible.

## Validation

Use the repository tooling after relevant changes:

1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run ty check src/terainet`
4. `uv run pytest -q`

Run only the applicable checks when a full suite is impractical, and report checks that could
not be run and why.

## Task-Specific Guidance

Load the relevant workflow skill for camera-trap data, TensorFlow/Keras training, notebook
extraction, or model release and inference work. Keep detailed procedures in those skills
rather than duplicating them here.
