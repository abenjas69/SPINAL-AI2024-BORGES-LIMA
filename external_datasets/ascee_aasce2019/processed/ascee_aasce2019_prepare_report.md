# AASCE 2019 dataset preparation

## Scope

Prepared the local real-radiograph dataset for evaluation-only use.
No training split, tuning, or model selection is performed here.

## Counts

- images: `481`
- filenames rows: `481`
- angle rows: `481`
- landmark rows: `481`
- per-image mat files: `481`

## Image dimensions

- unique sizes: `474`
- width min/max: `355` / `1427`
- height min/max: `973` / `3755`

## Landmark audit

- `landmarks.csv` is parsed as `x1..x68,y1..y68`.
- Every 4 points are grouped into one vertebral quadrilateral.
- This gives 17 vertebrae per image.
- landmark geometry vs max label abs delta mean: `2.3069125602788687`
- landmark geometry vs max label abs delta p90: `4.276362485605361`

## Artefacts

- manifest JSONL: `external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.jsonl`
- manifest CSV: `external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.csv`
- summary JSON: `external_datasets/ascee_aasce2019/processed/ascee_aasce2019_summary.json`
