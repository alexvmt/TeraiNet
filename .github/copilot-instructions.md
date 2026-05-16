# GitHub Copilot Instructions

## Project Context

This repository contains machine learning and computer vision workflows related to wildlife conservation, species classification, and camera trap imagery, with a focus on the Terai region in Nepal and Bengal tiger conservation.

Primary goals of this repository:
- Convert exploratory notebooks into maintainable Python scripts
- Improve modularity incrementally
- Keep experiments reproducible
- Maintain readability and simplicity
- Avoid unnecessary abstractions and overengineering

This is a lightweight ML repository, not an enterprise framework.

The repository currently uses:
- Python 3.10
- TensorFlow 2.16
- Keras 3
- Polars and Pandas
- Ruff
- ty
- pytest
- notebook-heavy experimentation workflows

The repository structure follows a `src/` layout.

---

# General Engineering Philosophy

Prioritize:
1. Correctness
2. Preserving existing behavior
3. Readability
4. Simplicity
5. Reproducibility
6. Performance optimization only when necessary

Prefer:
- Explicit code over clever code
- Small focused functions
- Incremental refactors
- Minimal diffs
- Functional approaches where practical
- Simple module structures

Avoid:
- Premature optimization
- Excessive abstraction
- Deep inheritance hierarchies
- Enterprise patterns
- Unnecessary classes
- Overly generic utilities
- Hidden side effects
- Large "utils.py" dumping grounds

When uncertain:
- Prefer the simplest implementation
- Ask for clarification instead of inventing architecture

---

# Refactoring Guidelines

Most work in this repository involves:
- notebook -> script conversion
- cleanup
- modularization
- improving maintainability

When refactoring:
- Preserve behavior exactly unless explicitly instructed otherwise
- Do not silently change algorithms
- Do not silently change hyperparameters
- Keep outputs and file formats stable
- Keep public APIs stable unless requested
- Minimize diff size
- Refactor incrementally
- Prefer extraction over rewriting

Avoid introducing:
- framework-style abstractions
- unnecessary config systems
- unnecessary OOP
- unnecessary dependency injection

---

# Notebook Conversion Rules

Jupyter notebooks often contain hidden state and implicit execution order.

When converting notebooks into scripts:
- Remove implicit notebook state
- Make inputs and outputs explicit
- Convert cells into reusable functions
- Avoid global variables
- Avoid mutable shared state
- Separate data loading, preprocessing, training, evaluation, and inference
- Preserve reproducibility
- Preserve existing behavior

Prefer this structure:

```text
src/
    data/
    features/
    models/
    training/
    inference/
    evaluation/
    utils/
```

Avoid giant monolithic scripts.

---

# Python Best Practices

Use:
- Python 3.10 compatible syntax only
- Type hints
- pathlib instead of os.path
- dataclasses for simple configs
- logging instead of print for scripts
- clear function boundaries
- explicit return values
- descriptive variable names

Prefer:
- pure functions where practical
- early validation
- guard clauses over deep nesting
- composition over inheritance

Write concise docstrings for:
- public functions
- classes
- modules with non-obvious behavior

Avoid:
- deeply nested logic
- duplicated code
- magic numbers
- hidden mutations
- overly long functions

---

# Ruff / Formatting Rules

This repository uses Ruff with:
- line length = 100
- target-version = py310

Ensure generated code:
- passes Ruff checks
- uses sorted imports
- avoids unused imports
- avoids unnecessary complexity

Prefer:
- readable line breaks
- explicit imports
- small focused functions

Avoid wildcard imports.

---

# Type Checking Rules

This repository uses `ty`.

Generated code should:
- include practical type hints
- avoid unnecessary Any usage
- use explicit return types where useful
- keep typing readable and pragmatic

Do not overcomplicate typing.

---

# Testing Rules

This repository uses pytest.

When modifying code:
- preserve existing outputs
- validate shapes and datatypes
- validate training still works
- validate inference still works

Prefer:
- lightweight tests
- sanity checks
- assertions for critical assumptions

Add validation checks for:
- tensor shapes
- label mappings
- file existence
- configuration validity

---

# Data Science Best Practices

Ensure:
- reproducibility
- deterministic preprocessing where possible
- clear train/validation/test separation
- no data leakage
- explicit random seeds

Prefer:
- modular preprocessing pipelines
- explicit feature engineering
- clear experiment tracking
- reusable transformation functions

Avoid:
- hidden preprocessing steps
- modifying datasets in place unless intentional
- duplicated preprocessing logic between training and inference

When handling datasets:
- validate schema assumptions
- validate shapes and dtypes
- handle missing values explicitly
- preserve metadata when useful

Prefer Polars for:
- tabular transformations
- efficient data pipelines
- lazy operations where appropriate

Use Pandas only when:
- library compatibility requires it
- functionality is missing in Polars

---

# Machine Learning Best Practices

Prefer:
- reproducible experiments
- configurable hyperparameters
- clear separation between training and inference
- modular training loops
- explicit evaluation pipelines

Training code should:
- support checkpointing
- support resuming training where practical
- log metrics clearly
- fail loudly on invalid states

Avoid:
- mixing training and inference logic
- hidden augmentation behavior
- hardcoded paths
- hardcoded device assumptions

When adding new models:
- keep interfaces consistent
- separate architecture definition from training logic
- avoid unnecessary abstraction layers

---

# Computer Vision Best Practices

For image pipelines:
- preserve aspect ratio when appropriate
- document augmentations clearly
- ensure deterministic validation transforms
- separate train and inference transforms

When working with camera trap data:
- assume class imbalance
- assume noisy labels may exist
- preserve image metadata where useful

Prefer:
- tf.data pipelines
- efficient caching and prefetching
- vectorized preprocessing
- reusable transform functions

Avoid:
- hidden image conversions
- implicit channel assumptions
- inconsistent normalization pipelines

---

# TensorFlow / Keras Guidelines

This repository primarily uses:
- TensorFlow 2.16
- Keras 3
- kimm models

Prefer:
- tf.data pipelines
- reusable preprocessing functions
- configurable training loops
- mixed precision where appropriate
- deterministic validation pipelines

Avoid:
- unnecessary custom training loops when fit() is sufficient
- excessive callback complexity
- hidden device transfers
- hardcoded GPU assumptions

When debugging:
- validate tensor shapes explicitly
- validate label mappings
- validate augmentations visually where relevant

Prefer:
- modular model-building functions
- explicit input shapes
- configurable augmentation pipelines
- separating model definition from training configuration

---

# File and Repository Organization

Prefer:
- clear module boundaries
- one responsibility per file
- descriptive filenames
- configuration separated from logic

Avoid:
- large scripts with multiple responsibilities
- circular imports
- deeply nested packages
- excessive utility modules

Keep repository structure simple and pragmatic.

---

# Dependency Management

Prefer:
- minimal dependencies
- well-maintained libraries
- standard ML ecosystem tools

Avoid adding dependencies unless they provide clear value.

Before adding a dependency:
- check if functionality already exists in the standard library
- check if functionality already exists in TensorFlow/Keras ecosystem
- prefer lightweight solutions

---

# Logging and Experiment Tracking

This repository uses:
- wandb
- tqdm

Prefer:
- clear metric logging
- reproducible experiment naming
- explicit configuration logging
- progress bars for long-running operations

Avoid:
- excessive console spam
- hidden logging side effects

---

# Copilot / Claude Haiku Specific Instructions

This repository is intentionally small and pragmatic.

The user usually:
- already knows the intended architecture
- wants incremental improvements
- wants precise transformations
- does not want speculative redesigns

Therefore:
- Do not invent architecture
- Do not overengineer
- Do not introduce unnecessary abstractions
- Prefer minimal, targeted changes
- Preserve behavior unless explicitly instructed otherwise

For requests:
- Follow instructions literally
- Avoid making unrelated improvements
- Avoid speculative optimizations
- Avoid broad rewrites

For notebook-to-script conversion:
- Focus on extracting reusable functions
- Make state explicit
- Keep logic identical unless requested otherwise

For refactoring:
- Prefer incremental extraction/refinement
- Keep diffs small
- Avoid changing behavior

When generating code:
- Produce production-quality Python
- Include type hints where practical
- Include concise docstrings
- Prefer readability over cleverness

If requirements are ambiguous:
- ask clarifying questions
- do not invent complex solutions

---

# Preferred Project Characteristics

This repository should remain:
- lightweight
- understandable
- modular
- reproducible
- easy to iterate on
- practical for ML experimentation

The goal is maintainable research/engineering code, not framework development.
