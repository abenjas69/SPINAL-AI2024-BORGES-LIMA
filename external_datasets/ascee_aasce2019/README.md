# AASCE 2019 Evaluation Data

This folder contains the processed AASCE 2019 real-radiograph evaluation data
used in the final project analysis.

```text
raw/train/
  real radiographs

raw/train_txt/
  original landmark text files

processed/
  ascee_aasce2019_manifest.jsonl
  ascee_aasce2019_manifest.csv
  ascee_aasce2019_summary.json
```

The folder name keeps the historical `ascee_aasce2019` path used by the project
scripts, but the dataset is referred to as AASCE in reports and documentation.

Use the root runner for evaluation:

```powershell
python run_eval.py aasce-fusion --num-images 8
python run_eval.py aasce-fusion --num-images 0
```
