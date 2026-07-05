# Fusion Centerline + MLP V3

## Setup

- mode: `apply-locked`
- mlp predictions: not redistributed in this public portfolio tree
- centerline predictions: not redistributed in this public portfolio tree
- common rows: `3988`
- mlp rows: `3988`
- centerline rows: `3988`
- missing centerline rows: `0`
- missing mlp rows: `0`
- centerline bias correction: `5.587282` deg
- centerline weight: `0.2900`
- mlp weight: `0.7100`
- selection objective: `mae`
- lock source: `holdout`
- fusion improved rows: `2171`
- fusion rescued >5: `104`
- fusion broken >5: `33`

## Metrics

| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_only | 3988 | 2.345 | 5.3120% | 0.771 | 0.915 | 0.976 | 337 | 95 | 3.817 | 4.627 | 0.396 |
| centerline_only | 3988 | 6.397 | 20.2466% | 0.159 | 0.385 | 0.870 | 2454 | 517 | 7.456 | 10.655 | -5.665 |
| centerline_bias_corrected | 3988 | 3.051 | 7.1079% | 0.640 | 0.854 | 0.970 | 583 | 120 | 4.848 | 5.863 | -0.078 |
| fusion_v3_locked | 3988 | 2.214 | 5.0215% | 0.797 | 0.933 | 0.977 | 266 | 90 | 3.716 | 4.181 | 0.259 |

## Outputs

- predictions CSV: not redistributed in this public portfolio tree
- metrics JSON: `.\experiments\fusion_centerline_mlp_v3_final_test\fusion_centerline_mlp_v3_metrics.json`
- lock JSON: `.\experiments\fusion_centerline_mlp_v3_final_test\fusion_centerline_mlp_v3_lock.json`

## Notes

- In apply-locked mode, this script does not fit bias or sweep weights on the input CSVs.
- For the final test set, use the lock JSON created on holdout.
