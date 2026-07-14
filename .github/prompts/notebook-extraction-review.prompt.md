---
name: Notebook Extraction Review
description: "Review a TeraiNet notebook section and plan or implement safe extraction into tested reusable source code."
argument-hint: "Notebook path and cells or behavior to extract"
agent: agent
---

Use the `notebook-extraction` skill to review the requested notebook behavior.

Identify hidden state, inputs, outputs, configuration, filesystem effects, preprocessing, label
mapping, and seed dependencies. Search existing `src/terainet/` modules before proposing a new
module. Produce a minimal extraction plan, or implement it when explicitly requested, with tests
and a notebook-parity verification. Keep exploratory visualization in the notebook.
