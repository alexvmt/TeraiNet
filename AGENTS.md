# TeraiNet Agent Guide

This repository is an experimental machine learning project for wildlife
species classification using camera trap imagery.

The repository is intentionally lightweight and should remain easy to
understand and modify.

# Primary Goal

Make the smallest change that correctly solves the requested task.

Preserve existing behavior unless explicitly instructed otherwise.

# Planning

Before making changes:

1. Identify the minimal affected files.
2. Understand existing implementation patterns.
3. Reuse existing code where possible.
4. Avoid introducing new abstractions unless necessary.

# Refactoring Rules

Prefer:

- extraction over rewriting
- incremental changes over redesigns
- composition over inheritance
- existing project patterns over new patterns

Avoid:

- speculative abstractions
- framework-style architectures
- dependency injection frameworks
- deep inheritance hierarchies
- large utility modules

# Notebook Conversion Workflow

When converting notebook code:

1. Remove hidden notebook state.
2. Make inputs and outputs explicit.
3. Extract reusable functions.
4. Separate loading, preprocessing, training, evaluation, and inference.
5. Preserve existing outputs.
6. Add tests where practical.

# ML Safety Rules

- Prevent data leakage.
- Keep train, validation, and test logic separate.
- Preserve label mappings.
- Preserve preprocessing consistency between training and inference.
- Preserve existing model behavior unless instructed otherwise.

# Validation

Before finalizing changes, run:

```bash
ruff check .
ruff format .
ty check
pytest
```

If a command cannot be executed, explain why.

# Scope Control

Do not modify unrelated files.

Do not rewrite working code without a clear benefit.

Do not introduce new dependencies without justification.

Prefer small, reviewable diffs.
