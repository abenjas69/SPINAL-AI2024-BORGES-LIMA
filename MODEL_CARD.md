# Model Card

## Model Family

SPINAL-AI2024-BORGES-LIMA contains three final inference components:

- Landmark pipeline: vertebral landmark detection, anatomical post-processing,
  Cobb geometry, and residual MLP calibration.
- Centerline pipeline: centerline extraction and tangent/derivative-based angle
  estimation.
- Locked late fusion: a fixed weighted average of landmark and centerline Cobb
  estimates.

## Intended Use

The repository is intended for academic review, reproducibility checks,
portfolio assessment, and research discussion around interpretable scoliosis
measurement from radiographs.

The public repository supports aggregate-metric inspection without datasets.
Full image-level reruns require local restoration of the upstream datasets and
annotations described in `DATASET_ACCESS.md`.

## Out Of Scope

The models are not intended for:

- autonomous diagnosis;
- clinical decision-making without expert review;
- deployment in medical workflows;
- use on patient populations or image domains not validated by a clinician-led
  study.

## Inputs

AP/PA spinal radiographs with enough visible vertebral structure to estimate a
spinal curve.

## Outputs

Estimated Cobb angle values in degrees, intermediate geometric predictions, and
optional visual overlays showing landmarks, centerline, and selected Cobb
signals.

## Evaluation Summary

| Evaluation | N | MAE | Within 5 deg |
|---|---:|---:|---:|
| Landmark MLP v2, subset5 | 3988 | 2.3448 deg | 91.55% |
| Centerline bias-corrected, subset5 | 3988 | 3.0513 deg | 85.38% |
| Locked fusion v3, subset5 | 3988 | 2.2140 deg | 93.33% |
| AASCE fusion zero-shot | 323 covered / 481 total | 18.4423 deg | 15.79% |
| AASCE GT-landmark geometry audit | 481 | 2.3069 deg | 95.63% |

## Known Limitations

- Domain shift from synthetic/benchmark images to real radiographs causes a
  major performance drop.
- Very high Cobb angles and severe deformations are harder because vertebrae can
  be rotated, partially visible, or outside the training distribution.
- Landmark errors propagate into endplate selection and Cobb estimation.
- Centerline estimates can be biased when the predicted curve follows the wrong
  anatomical signal.

## Ethical And Safety Notes

The project works with medical-image data. Dataset licensing, privacy,
institutional rules, and patient-data restrictions must be respected before any
reuse. Outputs should be reviewed as research measurements, not diagnoses.
