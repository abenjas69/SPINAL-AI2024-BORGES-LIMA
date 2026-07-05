# Dataset Access

This repository does not mirror the datasets used by the project. It keeps code,
trained weights, locked configuration, aggregate metrics, and documentation.

The reason is practical and professional: public access to a dataset does not
automatically grant third-party redistribution rights from this repository.

## SPINAL-AI2024

Upstream source:

```text
https://github.com/Ernestchenchen/Spinal-AI2024
```

The upstream repository publicly hosts the generated Spinal-AI2024 dataset and
describes 20,000 generated scoliosis X-ray images split into five subsets. The
authors recommend subset1-subset4 for training and subset5 for test/evaluation.

At the time this repository was prepared, the upstream dataset repository did
not expose an explicit license file. For that reason, this project references
the upstream source instead of republishing the images or complete annotations.

Expected local placement for full subset5 reruns:

```text
raw/images/test/Spinal-AI2024-subset5/
processed/cleaned/test_ready_annotations_clean.json
```

## AASCE 2019

AASCE 2019 is a real-radiograph benchmark used for zero-shot domain-transfer
analysis in this project. Because it contains medical images and no clear
third-party redistribution permission was identified, this repository keeps only
aggregate results and preparation summaries.

Expected local placement for full AASCE reruns:

```text
external_datasets/ascee_aasce2019/raw/
external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.jsonl
```

Use this dataset only if you have legitimate access to the original files and
permission to use them for your purpose.

## Raw-Data-Free Repository Check

The repository keeps aggregate metrics that can be inspected without datasets:

```powershell
python run_eval.py metrics-summary
```

Image-level commands such as `subset5-diogo`, `subset5-daniel`,
`subset5-fusion`, and `aasce-fusion` require the local dataset artefacts above.
