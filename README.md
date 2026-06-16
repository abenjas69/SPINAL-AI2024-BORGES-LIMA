# SPINAL-AI2024 Borges Evaluation Pack

Clean evaluation repository for the scoliosis project by Diogo Borges and
Daniel Lima.

This package lets the evaluator clone one small, focused repository and test:

- Diogo Borges' final landmark/Cobb pipeline on SPINAL-AI2024 subset5.
- Daniel Lima's centerline model on the same subset.
- The locked fusion between Diogo's MLP-Cobb prediction and Daniel's centerline prediction.
- Zero-shot behavior on AASCE 2019 real radiographs.

The original research repository contains many historical experiments. This
repository intentionally keeps only the files needed for assessment.

## Data Included

| Dataset | Content | Purpose |
|---|---:|---|
| SPINAL-AI2024 subset5 | 4000 test images, 3988 valid annotated samples | Official synthetic/benchmark test subset |
| AASCE 2019 | 481 real radiographs with processed manifest | Real-data generalization test |

The SPINAL-AI2024 subset is the controlled project benchmark. AASCE is a real
radiographic dataset and should be interpreted as a domain-shift/generalization
test, not as a retrained or tuned result.

## Repository Layout

```text
models/
  phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras
  phase9_cobb_residual_mlp_v2.keras
  phase9_cobb_residual_mlp_v2_scaler.npz
  centerline_daniel_unet_baseline_2000_padding_512.keras

processed/cleaned/
  test_ready_annotations_clean.json

raw/images/test/Spinal-AI2024-subset5/
  016001.jpg ... 020000.jpg

external_datasets/ascee_aasce2019/
  raw/
  processed/ascee_aasce2019_manifest.jsonl

scripts/
  numbered scripts needed by the final locked evaluations

experiments/reference/
  stored reference outputs from the final project runs

run_eval.py
  simple command wrapper for the evaluator
```

Large model weight files are tracked with Git LFS.

## Setup

Use Python 3.10 or 3.11 if possible.

```powershell
git clone https://github.com/abenjas69/SPINAL-AI2024-BORGES-EVAL.git
cd SPINAL-AI2024-BORGES-EVAL
git lfs pull

python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Git LFS was not installed before cloning, install it and run:

```powershell
git lfs install
git lfs pull
```

## Fallback Without Git LFS

If `git lfs pull` fails because the GitHub LFS bandwidth quota is exhausted,
clone the repository without LFS smudge and download the model weights from the
GitHub Release asset instead:

```powershell
$env:GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/abenjas69/SPINAL-AI2024-BORGES-EVAL.git
cd SPINAL-AI2024-BORGES-EVAL
```

Then download `spinal-ai2024-borges-eval-models-v1.zip` from:

```text
https://github.com/abenjas69/SPINAL-AI2024-BORGES-EVAL/releases/tag/models-v1
```

Extract the zip into the repository root, replacing the LFS pointer files:

```powershell
Expand-Archive ..\spinal-ai2024-borges-eval-models-v1.zip -DestinationPath . -Force

python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_eval.py all-smoke --num-images 2
```

If a `.keras` file starts with `version https://git-lfs.github.com/spec/v1`,
it is still only an LFS pointer and the zip was not extracted correctly.

## Quick Smoke Test

This evaluates a tiny window to check that models, data paths, and scripts load.

```powershell
python run_eval.py all-smoke --num-images 2
```

Outputs are written under `outputs/`.

## Main Commands

Run Diogo's final model on a small subset:

```powershell
python run_eval.py subset5-diogo --num-images 8
```

Run Daniel's centerline model on a small subset:

```powershell
python run_eval.py subset5-daniel --num-images 8
```

Run the full locked fusion on a small subset:

```powershell
python run_eval.py subset5-fusion --num-images 8
```

Apply the locked fusion to the stored full subset5 predictions:

```powershell
python run_eval.py subset5-fusion-reference
```

Run the AASCE real-data fusion test on a small subset:

```powershell
python run_eval.py aasce-fusion --num-images 8
```

For full evaluations, pass `--num-images 0`:

```powershell
python run_eval.py subset5-diogo --num-images 0
python run_eval.py subset5-daniel --num-images 0
python run_eval.py subset5-fusion --num-images 0
python run_eval.py aasce-fusion --num-images 0
```

Full runs can be slow on CPU because they load TensorFlow models and process
thousands of images.

## Reference Results

The reference outputs are stored in `experiments/reference/`.

| Evaluation | Images | MAE | SMAPE | within 5 deg |
|---|---:|---:|---:|---:|
| Diogo MLP v2, subset5 | 3988 | 2.3448 deg | 5.3120% | 91.55% |
| Daniel centerline, subset5 max Cobb | 3988 | 6.3969 deg | 20.2466% | 38.47% |
| Fusion v3 locked, subset5 | 3988 | 2.2140 deg | 5.0215% | 93.33% |
| AASCE fusion, real zero-shot | 323 covered / 481 total | 18.4423 deg | 26.5564% | 15.79% on covered rows |
| AASCE GT-landmark geometry audit | 481 | 2.3069 deg | 3.5775% | 95.63% |

The AASCE result is deliberately reported as zero-shot domain transfer. The
landmark-geometry audit shows that Cobb computation is valid when the landmarks
are reliable, while the model predictions degrade under real-domain shift.

## Important Notes

- The project estimates Cobb angles from radiographs and is for academic
  analysis, not autonomous diagnosis.
- The fusion lock was selected on the internal holdout split and then applied
  unchanged to subset5 and AASCE.
- Ground truth labels are used only for metric computation, not for ROI
  selection or inference.
- AASCE is a real-image dataset; SPINAL-AI2024 subset5 is the project benchmark
  test subset.

## Troubleshooting

If a model file is only a tiny pointer file, Git LFS did not download the real
weights. Run:

```powershell
git lfs pull
```

If TensorFlow cannot be installed, check the Python version first. Python 3.10
or 3.11 is the safest option for this package.
