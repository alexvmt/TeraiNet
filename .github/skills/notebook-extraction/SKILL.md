---
name: notebook-extraction
description: "Use when converting Jupyter notebook experiments into reusable TeraiNet source code, removing hidden notebook state, extracting preprocessing/training/inference helpers, or adding tests for notebook logic."
---

# Notebook Extraction

## When to Use

- moving repeatable notebook logic into `src/terainet/`
- making notebook preprocessing, training, evaluation, or inference reusable
- removing reliance on prior cell execution, globals, or hard-coded working directories

## Extraction Procedure

1. Identify the smallest self-contained behavior to extract and list its inputs, outputs,
   filesystem effects, configuration values, and random state.
2. Search `src/terainet/` for an existing home or compatible utility before creating a module.
3. Extract pure transformations first. Use a class only when durable state is required.
4. Make paths, configuration, class mappings, preprocessing, and seeds explicit parameters.
5. Add focused tests under `tests/` before replacing notebook behavior.
6. Update the notebook to import and call the extracted API; retain exploration and visualization
   in the notebook.
7. Compare the previous and extracted outputs on representative data, including failure cases.

## Boundaries

- Preserve dataset split boundaries and label ordering.
- Do not copy notebook globals, credentials, environment-specific paths, or interactive setup
  into library code.
- Avoid creating broad utility modules. Place code with its closest domain: data, detection,
  models, or a narrowly named new module.
- Keep public APIs small, typed where practical, and documented with behavior and errors.

## Completion Checklist

- [ ] No extracted function depends on prior notebook-cell state.
- [ ] Inputs, outputs, paths, configuration, and seed are explicit.
- [ ] Tests cover normal behavior and meaningful invalid input.
- [ ] Notebook behavior is preserved using the imported implementation.
- [ ] Relevant linting, type checking, and tests pass.
