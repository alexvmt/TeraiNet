# Repository Overview

This repository contains machine learning and computer vision workflows for:
- wildlife conservation
- species classification
- camera trap imagery
- Bengal tiger conservation in the Terai region of Nepal

The repository is intentionally lightweight and experimentation-focused.

Canonical workflow:
1. exploratory notebook experimentation
2. reusable preprocessing extraction
3. modular TensorFlow/Keras training pipelines
4. evaluation and inference workflows
5. gradual notebook → script refactoring

# Highest Priority Constraints

- Preserve existing behavior unless explicitly instructed otherwise.
- Keep changes incremental and minimal.
- Avoid unnecessary abstractions and framework-style architecture.
- Keep preprocessing consistent between training and inference.
- Prevent data leakage.
- Keep experiments reproducible.
- Reusable logic must not remain notebook-only.

# Repository Characteristics

This repository should remain:
- lightweight
- modular
- understandable
- reproducible
- practical for ML experimentation

This is not an enterprise framework repository.

# Repository Structure

- `src/` → reusable Python modules
- `notebooks/` → experimentation and exploration
- `tests/` → pytest tests
- `configs/` → configuration files
- `results/` → generated artifacts

# Anti-Patterns

- Do not silently change algorithms or hyperparameters.
- Do not introduce speculative architecture.
- Do not introduce unnecessary OOP or dependency injection.
- Do not create large generic utility modules.
- Do not duplicate preprocessing logic between training and inference.
- Do not mix training and inference logic.
- Do not hardcode paths or device assumptions.

# Tooling

Before committing changes:

```bash
ruff check .
ruff format .
ty check
pytest
```

See `.github/skills/` for workflow-specific guidance.
