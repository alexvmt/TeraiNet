---
name: model-release-inference
description: "Use when packaging a TeraiNet model release, building or reviewing inference code, exporting class mappings, validating model artifacts, or preparing deployment to Hugging Face or AddaxAI."
---

# Model Release and Inference

## Release Bundle

Package a retained model with all information required to load and use it without training data:

- model file and format;
- ordered class-list artifact;
- effective configuration and preprocessing contract;
- source data and split provenance;
- seed, framework dependency versions, and checkpoint-selection rationale;
- validation, test, and out-of-distribution metrics reported separately.

Use `terainet.models.create_class_list_yaml_file()` to create the class-list artifact rather
than recreating label-index logic in a notebook or deployment script.

## Inference Verification

1. Load the model and class-list artifact in a clean process with no training dataset access.
2. Validate input image mode, resize, normalization, channel order, and batch shape against the
   training preprocessing contract.
3. Confirm each output index maps to the saved ordered class list.
4. Run deterministic predictions on representative held-out examples and record results.
5. Treat missing or incompatible metadata as a release blocker, not a warning.

## Deployment Boundaries

- Do not expose credentials, access tokens, or local filesystem assumptions in release code.
- Do not claim generalization from validation data or merge OOD results into test metrics.
- Version model artifacts and configuration together; never overwrite a published artifact in
  place without an explicit new release identifier.

## Completion Checklist

- [ ] Bundle contains model, mapping, configuration, preprocessing contract, and metrics.
- [ ] Model-load and inference smoke test passed in a clean environment.
- [ ] Prediction indices match the saved class list.
- [ ] Test and OOD outcomes are clearly labelled.
