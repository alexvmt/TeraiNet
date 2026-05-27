---
name: camera-trap-ml-workflow
description: Wildlife camera trap classification workflows and dataset handling conventions.
---

# When To Use

Use this skill when:
- handling wildlife image datasets
- implementing species classification workflows
- implementing augmentation pipelines
- working with camera trap imagery
- handling label mappings or metadata

# When Not To Use

Do not use this skill for:
- TensorFlow architecture implementation details
- generic repository refactoring
- experiment tracking setup only

# Highest Priority Constraints

- Assume class imbalance exists.
- Preserve label consistency across workflows.
- Keep preprocessing deterministic for validation and inference.
- Preserve metadata when useful.

# Canonical Workflow

camera trap dataset
→ validation and schema checks
→ preprocessing
→ augmentation
→ train/validation/test split
→ training
→ evaluation
→ inference

# Dataset Rules

- Validate dataset schemas explicitly.
- Validate image paths before training.
- Handle missing values explicitly.
- Keep train/validation/test separation explicit.
- Preserve label mappings consistently.

# Augmentation Rules

- Keep validation augmentations deterministic.
- Separate train and inference transforms.
- Document augmentations clearly.
- Preserve aspect ratio when appropriate.

# Evaluation Rules

- Inspect per-class metrics.
- Validate minority-class performance explicitly.
- Check confusion matrices for ecologically similar species.
- Validate label mapping consistency.

# Anti-Patterns

- Do not apply random augmentations during validation.
- Do not mix label mappings between datasets.
- Do not assume balanced datasets.
- Do not silently drop failed image loads.
- Do not duplicate preprocessing logic.
