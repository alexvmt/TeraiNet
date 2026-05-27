---
name: lightweight-refactoring-workflow
description: Incremental notebook-to-script conversion and lightweight ML repository refactoring workflows.
---

# When To Use

Use this skill when:
- converting notebooks into scripts
- modularizing workflows
- extracting reusable functions
- reorganizing repository structure
- improving maintainability

# When Not To Use

Do not use this skill for:
- large architectural redesigns
- enterprise framework patterns
- speculative abstractions

# Highest Priority Constraints

- Preserve existing behavior.
- Keep diffs minimal.
- Refactor incrementally.
- Avoid unnecessary abstractions.
- Keep repository structure simple.

# Canonical Refactoring Workflow

exploratory notebook
→ explicit inputs/outputs
→ reusable functions
→ modular scripts
→ lightweight validation tests

# Core Rules

- Prefer extraction over rewriting.
- Keep public APIs stable unless instructed otherwise.
- Keep algorithms and hyperparameters unchanged unless instructed otherwise.
- Separate loading, preprocessing, training, evaluation, and inference logic.
- Use type hints where practical.

# Preferred Repository Characteristics

The repository should remain:
- lightweight
- understandable
- modular
- experimentation-friendly
- easy to iterate on

# Preferred Patterns

Prefer:
- small focused functions
- composition over inheritance
- functional approaches where practical
- explicit state handling
- simple module boundaries

# Anti-Patterns

- Do not introduce dependency injection frameworks.
- Do not introduce unnecessary config systems.
- Do not introduce deep inheritance hierarchies.
- Do not create giant `utils.py` files.
- Do not rewrite stable code unnecessarily.
- Do not convert everything into classes.

# Notebook Conversion Workflow

1. Remove implicit notebook state.
2. Make inputs and outputs explicit.
3. Extract reusable functions.
4. Separate pipeline stages.
5. Validate outputs remain unchanged.
6. Add lightweight tests where appropriate.
