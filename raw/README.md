# Raw Dataset Placement

Raw SPINAL-AI2024 radiographs and complete cleaned annotations are not
redistributed in this public repository.

To run the full image-level SPINAL-AI2024 subset5 evaluations, restore the
authorized local images under:

```text
raw/images/test/Spinal-AI2024-subset5/
```

Recommended setup from a local upstream clone:

```powershell
git clone --depth 1 https://github.com/Ernestchenchen/Spinal-AI2024.git ..\Spinal-AI2024
python scripts/prepare_spinal_ai2024_subset5.py --upstream ..\Spinal-AI2024
python run_eval.py check-data
```

Expected filenames follow the original subset5 range, for example:

```text
016001.jpg
016002.jpg
...
020000.jpg
```

This directory is ignored by Git so restored radiographs are not accidentally
committed.

The cleaned annotation file expected by the final evaluation scripts must also
be restored locally at:

```text
processed/cleaned/test_ready_annotations_clean.json
```
