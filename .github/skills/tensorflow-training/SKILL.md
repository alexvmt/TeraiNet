---
name: tensorflow-training-workflow
description: TensorFlow, Keras, and tf.data training workflows for wildlife image classification.
---

# When To Use

Use this skill when:
- implementing TensorFlow/Keras training pipelines
- modifying model architectures
- implementing tf.data pipelines
- implementing augmentations
- modifying training or evaluation workflows
- implementing checkpointing or logging

# When Not To Use

Do not use this skill for:
- repository refactoring decisions
- generic Python cleanup
- dataset schema validation only

# Highest Priority Constraints

- Preserve preprocessing consistency between training and inference.
- Prevent data leakage.
- Keep experiments reproducible.
- Keep validation deterministic.
- Preserve existing behavior unless explicitly instructed otherwise.

# Canonical Workflow

image loading
→ tf.data pipeline
→ preprocessing
→ augmentation
→ model training
→ validation
→ checkpointing
→ evaluation
→ inference

# Core Rules

- Use TensorFlow 2.16 and Keras 3 compatible APIs.
- Prefer tf.data pipelines.
- Separate training and inference transforms.
- Keep augmentation pipelines explicit.
- Validate tensor shapes explicitly.
- Set random seeds explicitly.

# Preferred Patterns

Prefer:
- reusable preprocessing functions
- configurable training functions
- modular model-building functions
- explicit input shapes
- checkpoint-based training
- mixed precision where appropriate

# Anti-Patterns

- Do not mix training and inference preprocessing.
- Do not hardcode GPU assumptions.
- Do not hide augmentations inside model code.
- Do not use unnecessary custom training loops when `fit()` is sufficient.
- Do not silently change normalization behavior.

# Debugging Workflow

1. Verify tensor shapes.
2. Verify label mappings.
3. Verify augmentation behavior.
4. Verify preprocessing consistency.
5. Verify deterministic validation transforms.
