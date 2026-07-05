# AASCE 2019 Evaluation Data

This folder keeps the processed AASCE 2019 manifests used by the final
real-domain evaluation.

```text
processed/
  ascee_aasce2019_manifest.jsonl
  ascee_aasce2019_manifest.csv
  ascee_aasce2019_summary.json
```

Raw AASCE radiographs and original landmark text files are not redistributed in
the active tree of this public repository. To rerun the full image-level AASCE
evaluation, obtain the dataset from its legitimate source and restore it
locally under:

```text
external_datasets/ascee_aasce2019/raw/
```

The folder name keeps the historical `ascee_aasce2019` path used by the project
scripts, but the dataset is referred to as AASCE in reports and documentation.

Use the root runner after restoring the data:

```powershell
python run_eval.py aasce-fusion --num-images 8
python run_eval.py aasce-fusion --num-images 0
```
