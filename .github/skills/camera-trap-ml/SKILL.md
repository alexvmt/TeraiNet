---
name: camera-trap-ml
description: "Use when working with camera-trap datasets, wildlife species labels, data sampling, location-aware splits, image filtering, or classifier evaluation. Preserves data integrity and prevents leakage."
---

# Camera-Trap ML Workflow

## When to Use

- working with camera trap datasets
- implementing species classification
- sampling or splitting image metadata
- handling label mappings or source-species substitutions
- filtering cropped/snipped images
- evaluating wildlife classifiers

## Required Data-Integrity Checks

1. Validate required schema columns, image paths, missing values, and duplicate identifiers
   before sampling or splitting.
2. Treat class names and their ordered mapping as a versioned contract. Validate that the
   observed classes match the configured classes before training and before interpreting
   predictions.
3. When `location_id` is available, split by location rather than image. Use
   `terainet.data.add_subset_column()` and verify with `check_location_split()`.
4. Do not move data between training, validation, test, or out-of-distribution subsets after
   model selection begins.
5. Record source datasets and any substitute species used for a target class. Do not silently
   broaden or relabel a class.

## Dataset Preparation

1. Validate metadata and select source images.
2. Split by acquisition location before sampling where location metadata exists.
3. For externally pre-split data, preserve its test subset; derive validation only from its
   training subset and document the decision.
4. Filter invalid or ambiguous examples explicitly. When using snippet crops, record the
   filtering rule and per-class retention rather than silently dropping files.
5. Create deterministic sampled datasets using an explicit seed.
6. Save the effective class mapping and dataset/split summary with the experiment artifacts.

## Training and Evaluation Boundaries

- Assume imbalance exists until class counts demonstrate otherwise.
- Keep augmentation in the training pipeline only. Validation, test, and inference transforms
  must be deterministic.
- Evaluate per-class precision, recall, F1, support, and a confusion matrix. Highlight
  minority classes and ecologically important target species.
- Report held-out test and out-of-distribution results separately; never combine them into one
  headline metric.
- Confirm label indices used for predictions match the saved class-list artifact.

## Completion Checklist

- [ ] Locations occur in one split only, when location data exists.
- [ ] Class mapping, sources, filtering, and seed are recorded.
- [ ] Training, validation, test, and OOD data remain distinct.
- [ ] Per-class and confusion-matrix results have been reviewed.
- [ ] Inference uses the same class mapping and deterministic preprocessing contract.
