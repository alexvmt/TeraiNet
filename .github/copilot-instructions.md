# Repository Overview

TeraiNet is a wildlife species classification project focused on
camera trap imagery from the Terai region of Nepal.

Primary use cases:

- species classification
- camera trap workflows
- conservation machine learning
- TensorFlow/Keras experimentation

Canonical workflow:

camera trap images
→ preprocessing
→ augmentation
→ TensorFlow training
→ evaluation
→ inference

# Repository Invariants

- Preserve preprocessing consistency between training and inference.
- Prevent data leakage.
- Preserve label mapping consistency.
- Keep experiments reproducible.
- Reusable logic belongs in `src/`, not notebooks.

# Repository Structure

- `src/` → reusable code
- `notebooks/` → experimentation
- `configs/` → configuration
- `tests/` → validation
- `results/` → generated artifacts

See `.github/skills/` for domain-specific workflows.
