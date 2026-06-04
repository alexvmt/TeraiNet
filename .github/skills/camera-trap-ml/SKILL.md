---
name: camera-trap-ml
description: Camera trap image classification and wildlife ML workflows.
---

# When To Use

Use this skill when:

- working with camera trap datasets
- implementing species classification
- handling label mappings
- creating augmentation strategies
- evaluating wildlife classifiers

# Highest Priority Constraints

- Assume class imbalance exists.
- Preserve label consistency.
- Preserve metadata when useful.
- Keep validation deterministic.

# Canonical Workflow

dataset validation
→ preprocessing
→ augmentation
→ train/validation/test split
→ training
→ evaluation
→ inference

# Dataset Rules

- Validate image paths.
- Validate dataset schemas.
- Handle missing values explicitly.
- Preserve label mappings.

# Evaluation Rules

- Inspect per-class metrics.
- Review confusion matrices.
- Evaluate minority species performance.
- Check for label mapping inconsistencies.

# Augmentation Rules

- Use augmentations only for training.
- Keep validation deterministic.
- Document augmentation assumptions.

# Anti-Patterns

- random validation augmentations
- changing label mappings mid-workflow
- assuming balanced datasets
- silently dropping failed images
