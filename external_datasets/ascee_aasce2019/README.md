# AASCE 2019 Evaluation Data

This folder keeps only aggregate preparation notes for the AASCE 2019
real-domain evaluation.

Raw AASCE radiographs, original landmark text files, and processed per-image
manifests are not redistributed in the active tree of this public repository.
To rerun the full image-level AASCE evaluation, obtain the dataset through a
legitimate source and restore the local artefacts under:

```text
external_datasets/ascee_aasce2019/raw/
external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.jsonl
```

The folder name keeps the historical `ascee_aasce2019` path used by the project
scripts, but the dataset is referred to as AASCE in reports and documentation.

Use the root runner after restoring the data and manifest:

```powershell
python run_eval.py aasce-fusion --num-images 8
python run_eval.py aasce-fusion --num-images 0
```
