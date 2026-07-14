---
name: tensorflow-training
description: "Use when modifying TensorFlow or Keras image-classification models, tf.data pipelines, augmentations, checkpoints, experiment tracking, or evaluation. Ensures reproducible training and inference consistency."
---

# TensorFlow and Keras Training

## When to Use

- modifying TensorFlow models
- implementing training pipelines
- implementing tf.data pipelines
- modifying augmentations
- implementing evaluation workflows
- saving checkpoints or experiment artifacts
- changing W&B experiment logging

## Compatibility and Reproducibility

- Treat [pyproject.toml](../../../pyproject.toml) as the source of truth for supported Python,
  TensorFlow, Keras, and related library versions.
- Use APIs compatible with those declared dependencies; do not duplicate version pins in code
  or documentation.
- Set and record an explicit seed for Python, NumPy, and TensorFlow before dataset shuffling or
  model construction. Record any remaining non-deterministic hardware behavior.
- Keep model input shapes, preprocessing parameters, class ordering, and dataset split metadata
  explicit in the run configuration.

## Workflow

1. Validate image/label tensors and the ordered class mapping.
2. Build a deterministic `tf.data` input pipeline with explicit shuffling and batching.
3. Apply shared, deterministic preprocessing for training, validation, test, and inference.
4. Apply stochastic augmentation only on the training path.
5. Train with explicit callbacks and checkpoint criteria.
6. Evaluate the selected checkpoint on validation and held-out data without augmentation.
7. Save the model with the effective configuration, class mapping, preprocessing contract, seed,
   metrics, and checkpoint-selection rationale.

## Experiment Tracking

- Use a descriptive run name that identifies model, dataset/version, and meaningful variant.
- Log hyperparameters, seed, data split summary, model artifact, class mapping, and final
  metrics together. Do not rely on hard-coded account, project, or entity settings.
- Version the effective configuration with each retained model. Never overwrite an artifact
  without recording its source run.

## Inference Consistency Check

Before releasing or changing inference code, verify that it loads the saved model and mapping,
uses deterministic preprocessing with the same resize/normalization/channel conventions, and
returns labels in saved index order. Smoke-test on held-out examples without access to training
data.

## Debugging Order

1. Verify tensor shapes.
2. Verify label mappings.
3. Verify augmentations.
4. Verify preprocessing and data ranges.
5. Verify validation transforms and checkpoint selection.
6. Verify the saved model loads and predicts with its recorded artifacts.
