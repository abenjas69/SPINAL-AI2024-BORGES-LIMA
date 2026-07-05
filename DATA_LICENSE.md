# Data And License Policy

This repository separates project code from medical-image data.

## Code

The source code in this repository is released under the MIT License, unless a
file states otherwise.

## Trained Models

The trained weights are included as research artifacts to reproduce the final
project evaluation. They are not medical devices and are not approved for
clinical use.

Before redistributing, publishing, or commercially using these weights, verify
that the training data terms, institutional rules, and any third-party dataset
licenses permit that use.

## Raw Radiographs

Raw radiographs, complete per-image annotations, processed dataset manifests,
and per-image prediction CSVs are not redistributed in the active tree of this
public portfolio repository.

To run image-level evaluations, obtain the datasets from their legitimate
sources and restore the expected local paths:

```text
raw/images/test/Spinal-AI2024-subset5/
processed/cleaned/test_ready_annotations_clean.json
external_datasets/ascee_aasce2019/raw/
external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.jsonl
```

These paths are ignored by Git so that restored radiographs are not accidentally
committed again.

## Metadata And Reference Outputs

The repository keeps aggregate metric JSON files, summary reports, and the
locked fusion configuration so that the final reported results can be inspected
without redistributing dataset-like per-image artefacts.

## History Note

This portfolio branch removes raw radiographs from the active repository tree.
If a stricter data takedown is required, perform a separate Git history and LFS
audit before treating the repository as fully scrubbed.
