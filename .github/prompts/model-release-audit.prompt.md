---
name: Model Release Audit
description: "Audit a TeraiNet model, checkpoint, or inference workflow for reproducibility and release readiness."
argument-hint: "Model artifact, notebook, or inference path to audit"
agent: agent
---

Use the `model-release-inference` skill to audit the requested model release or inference
workflow. Check that the release bundle is complete, class-index mapping is preserved,
preprocessing matches training, and test/OOD outcomes are reported separately. Return findings
ranked by release risk, the smallest corrective actions, and the validation evidence required.
