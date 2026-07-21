# Predicting Progression from MCI to Alzheimer's Disease

A reproduction and extension of **Minhas et al. (2017)** on the **ADNI** dataset:
predicting whether a patient with Mild Cognitive Impairment (MCI) will convert to
Alzheimer's Disease (AD), by **forecasting future biomarker values** and then
classifying conversion.

## Approach

1. **Feature ranking** — a two-sample Student's *t*-test on baseline biomarkers
   ranks each feature by how well it separates converters (MCIp) from stable
   patients (MCIs). Biomarker subsets are built by adding features in order of
   significance; the subset size = the "model rank". (`src/svm.py`)
2. **Biomarker forecasting** — future visits (24- and 30-month) are predicted
   from earlier visits with linear autoregression, using two schemes from the
   paper:
   - **SPM** (single-path): each biomarker is predicted from its **own** past values.
   - **MPTM** (multi-path): each past timepoint's full biomarker vector predicts
     the future value; the per-timepoint predictions are **averaged**.
3. **Classification** — an RBF-kernel **SVM** classifies conversion from the
   (real or forecast) biomarkers, evaluated with **5-fold leave-one-subject-out
   cross-validation** (`GroupKFold` on patient `RID`, so no subject leaks between
   train/test). Metrics: accuracy, AUC, sensitivity, specificity, precision.

Results match the paper closely and in several configurations exceed it.

## Data (ADNI — not included)

This project uses the **ADNI** dataset, which is governed by a Data Use
Agreement and **cannot be redistributed**, so the data CSVs are intentionally
**not** in this repo. Request access and download from
<https://adni.loni.usc.edu/>. The expected schema (see `report/Dataset Details.docx`):

| Column | Meaning |
| --- | --- |
| 1 | Serial number |
| 2 (`RID`) | Patient ID |
| 3 (`VISCODE`) | Visit: 1=baseline, 2=6mo, 3=12mo, 4=18mo, 5=24mo, 6=30mo |
| 4 (`Labels`) | 0 = stable (MCIs), 1 = converted to AD (MCIp) |
| 5–12 | Neuropsychological Measures (NM) |
| 13–268 | Imaging biomarkers (MRI) |

Place the raw file as `output_complete.csv` in the repo root before running.

## Repo structure

Only the code used in the final pipeline is included (the discarded lagged /
autoregressive experiments and scratch notebooks were left out).

```
src/
  svm.py            # 1. t-test feature ranking + incremental-subset SVM
  csvfile2.py       # 2. build reduced dataset (NM + 7 best MRI biomarkers)
  test3.py          # 3a. SPM & MPTM forecasting (full features)
  test4.py          # 3b. SPM & MPTM forecasting (reduced features)
  svm2.py           # 4a. classify on NM + MRI  -> results NM+MRI.csv
  svm_MRI.py        # 4b. classify on MRI only   -> results MRI.csv
  svm_NM.py         # 4c. classify on NM only    -> results NM.csv
results/            # final metric tables (accuracy / AUC / sens / spec / prec)
report/             # write-up + dataset schema
```

## Run order

```bash
pip install -r requirements.txt
# put the ADNI file as output_complete.csv in the repo root, then:
python src/svm.py          # rank features, find best biomarker subset
python src/csvfile2.py     # build new_output_data.csv (reduced features)
python src/test3.py        # forecast 24/30-month biomarkers (SPM + MPTM)
python src/svm2.py         # classify + write results
```

## Results (highlights)

Full tables are in `results/`. Using leave-one-subject-out 5-fold CV:

- **Accuracy up to ~0.88** and **AUC up to ~0.78**, comparable to and in places
  better than the published results.
- Forecast (SPM/MPTM) biomarkers classify almost as well as ground-truth future
  visits — i.e. the future visits can be usefully predicted from earlier ones.

## Reference

Minhas et al., *"Predicting Progression from Mild Cognitive Impairment to
Alzheimer's Disease Using Autoregressive Modelling of Longitudinal and
Multimodal Biomarkers"* (2017). IEEE JBHI. DOI:
[10.1109/JBHI.2017.2703918](https://doi.org/10.1109/JBHI.2017.2703918)
