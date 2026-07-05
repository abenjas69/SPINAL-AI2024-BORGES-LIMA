# Cleaned Annotation Placement

Complete cleaned SPINAL-AI2024 annotation JSON files are not redistributed in
this public portfolio repository.

To run full image-level subset5 evaluations, restore the cleaned test annotation
file locally at:

```text
processed/cleaned/test_ready_annotations_clean.json
```

The recommended way to rebuild it from the public upstream annotation zip and
Cobb ground-truth file is:

```powershell
git clone --depth 1 https://github.com/Ernestchenchen/Spinal-AI2024.git ..\Spinal-AI2024
python scripts/prepare_spinal_ai2024_subset5.py --upstream ..\Spinal-AI2024
python run_eval.py check-data
```

This file is ignored by Git so restored dataset annotations are not accidentally
committed.
