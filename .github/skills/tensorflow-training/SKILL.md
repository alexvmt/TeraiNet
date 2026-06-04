---
name: tensorflow-training
description: TensorFlow and Keras training workflows for image classification.
---

# When To Use

Use this skill when:

- modifying TensorFlow models
- implementing training pipelines
- implementing tf.data pipelines
- modifying augmentations
- implementing evaluation workflows

# Highest Priority Constraints

- Preserve preprocessing consistency.
- Prevent data leakage.
- Keep validation deterministic.
- Preserve label mappings.

# Canonical Workflow

image loading
→ tf.data pipeline
→ preprocessing
→ augmentation
→ training
→ validation
→ checkpointing
→ evaluation
→ inference

# Core Rules

- Use TensorFlow 2.16 compatible APIs.
- Use Keras 3 compatible APIs.
- Prefer tf.data pipelines.
- Keep training and inference transforms separate.
- Set random seeds explicitly.
- Validate tensor shapes.

# Preferred Patterns

- reusable preprocessing functions
- modular model builders
- checkpoint-based training
- explicit input shapes

# Anti-Patterns

- mixing training and inference transforms
- hidden augmentations
- hardcoded GPU assumptions
- unnecessary custom training loops

# Debugging Workflow

1. Verify tensor shapes.
2. Verify label mappings.
3. Verify augmentations.
4. Verify preprocessing.
5. Verify validation transforms.
