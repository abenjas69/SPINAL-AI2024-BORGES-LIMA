# Case Study: Automated Cobb Angle Estimation

## Problem

Manual Cobb angle measurement is clinically important for scoliosis assessment,
but it is time-consuming and sensitive to the vertebrae and endplates selected
by the observer.

The project goal was to build an interpretable AI pipeline that estimates Cobb
angles from spinal radiographs while exposing the geometric evidence used by the
model.

## Approach

The final system uses two complementary geometric signals.

The landmark branch detects vertebral quadrilaterals, orders them anatomically,
selects candidate endplates, computes Cobb geometry, and applies a residual MLP
calibrator to correct systematic residual errors.

The centerline branch predicts the global spine curve and estimates Cobb-related
angles from tangent and derivative behavior along that curve.

The final prediction uses a locked late-fusion rule selected on an internal
holdout split:

```text
F = 0.71 * LandmarkMLP + 0.29 * CenterlineCorrected
```

This design preserves interpretability: the prediction can be traced back to
vertebral endpoints, endplate lines, the centerline, and the fixed fusion rule.

## Results

On SPINAL-AI2024 subset5, the landmark MLP achieved 2.3448 degrees MAE and the
locked fusion improved this to 2.2140 degrees MAE, with 93.33% of predictions
within 5 degrees.

On AASCE 2019 real radiographs, the zero-shot fusion result degraded
substantially, showing the expected domain-shift limitation. A separate
GT-landmark geometry audit achieved 2.3069 degrees MAE, indicating that the Cobb
calculation itself is sound when reliable landmarks are available.

## Engineering Work

- Built final reproducible evaluation scripts and reference outputs.
- Integrated landmark and centerline predictions into a single locked fusion
  evaluator.
- Added qualitative overlays to inspect good cases, failure cases, rescue cases,
  and broken-fusion cases.
- Separated model errors from geometric Cobb-computation errors through a
  real-domain audit.

## Limitations

- The models are research artifacts and were not clinically validated.
- Real-domain transfer is limited without domain adaptation or real-image
  fine-tuning.
- Severe curves and partial/ambiguous vertebral visibility remain harder cases.
- Raw radiographs are intentionally not redistributed in this public portfolio
  repository.

## Next Steps

- Add a small open demo set or synthetic sample set that can be redistributed.
- Improve real-domain robustness with validated external training data.
- Add CI for raw-data-free reference checks.
- Package inference as a documented command-line interface or lightweight demo.
