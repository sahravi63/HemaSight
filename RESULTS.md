# HemaSight: model results

All models were trained and every number below was measured on the **IBM LinuxONE Community
Cloud** (s390x, Ubuntu 24.04, 3 CPUs), using only photos each model never saw during training.

## 1. Haemoglobin estimate from the inner eyelid (main model)

**Data:** 710 eyelid photos of children aged 6–59 months from 10 hospitals (CP-AnemiC), each paired
with a laboratory Hb value.

**How it is tested:** leave-one-hospital-out. The model is trained on 9 hospitals and tested on the
10th, and this is repeated for all 10, which shows how it would do at a clinic it has never seen.
The test photos are also re-coloured as if taken by a different phone camera (random per-channel
colour shifts of ±4% and ±8%).

| Model (trained on IBM Z) | Unseen hospital | + new camera ±4% | + new camera ±8% |
|---|---|---|---|
| **5-seed ensemble: Extra Trees + colour-jitter training (deployed)** | **AUC 0.930**, MAE 1.39 g/dL | **0.906** | **0.856** |
| Same model, single seed (average of 5 seeds) | AUC 0.928 ± 0.003, MAE 1.396 ± 0.011 | | |
| IBM Snap ML Random Forest | AUC 0.919, MAE 1.42 | 0.889 | 0.833 |
| Histogram Gradient Boosting | AUC 0.896, MAE 1.46 | 0.872 | 0.828 |
| Baseline: always guess the average Hb | MAE 1.81 | | |

AUC is for detecting anaemia (Hb < 11 g/dL).

**As a screening tool:** flagging every child whose estimate is below 10.68 g/dL catches **90%** of
anaemic children and correctly clears **79%** of healthy ones (seed ensemble).

Standard 5-fold CV with mixed hospitals gives a similar MAE of 1.42 g/dL and AUC of 0.911.

## 2. Design experiments (re-run on IBM Z with `experiments.py`)

| Approach | Unseen hospital | + new camera ±4% | + new camera ±8% |
|---|---|---|---|
| Colour features, plain training | AUC 0.939, MAE 1.34 | 0.816 | **0.671** |
| **Colour features + colour-jitter training (used)** | AUC 0.923, MAE 1.40 | **0.897** | **0.845** |
| Pretrained ResNet-18 features + SVR | AUC 0.892, MAE 1.45 | | |
| Guess the average | MAE 1.81 | | |

**What this shows:**
- The plain model looks best on familiar cameras but is **fragile**: a ±8% colour difference between
  phones drops it from AUC 0.94 to 0.67. Colour-jitter training gives up 0.016 AUC on familiar
  cameras and keeps 0.85 on a new one, so it is the model we deploy.
- A pretrained deep network (ResNet-18) is **worse** than colour features for this task. Pallor is a
  colour signal, and ImageNet networks are trained to ignore colour shifts.
- PyTorch has no build for IBM Z, so ResNet-18 runs in **pure numpy on s390x**
  (`hemasight/cnn.py`). It reads the official checkpoint directly and processes 710 photos in 158 s
  (about 0.2 s per photo on 3 CPUs).

## 3. Palm and fingernails (yes/no anaemia)

Grouped 5-fold CV on the full augmented datasets: all copies of a photo stay in the same fold, and
training uses colour jitter. Deployed models are **5-seed ensembles** (`train_seeds.py`); each seed
changes the fold split, the colour-jitter copies and the model's randomness.

| Site | Deployed model | AUC (single seeds) | **AUC (5-seed ensemble)** | New camera ±4% | ±8% | Specificity at 90% sensitivity | Role in the app |
|---|---|---|---|---|---|---|---|
| Palm | Random Forest | 0.859 ± 0.008 | **0.876** | 0.850 | 0.818 | 0.67 | Votes, weighted by accuracy |
| Fingernails | Extra Trees | 0.761 ± 0.019 | **0.775** | 0.739 | 0.722 | 0.42 | **Supporting only**; never decides alone |

In the single-seed comparison, IBM Snap ML Random Forest matched the best scikit-learn model on both
sites (palm 0.846, nails 0.723). The seed ensembles took 56 minutes to train on LinuxONE (3 CPUs).

## Known limitations

- Estimates are pulled toward the average. Severe anaemia (true Hb about 5.5) is estimated around 9,
  but the screening rule still flags almost all severe cases for a blood test.
- Trained only on young children in Ghana. Adults, pregnant women and other populations need local
  validation data before use.
- Colour shifts larger than ±8%, such as strong indoor lighting, still reduce accuracy. The printed
  colour-calibration card in the HemaSight design is the next step.
- Photos must be cut out to the tissue region, as in the training data. The app's outline tool does
  this, and un-outlined eyelid and nail photos are refused.
- The IBM Z AI accelerator (NNPA) was not exposed on this Community Cloud VM, so Snap ML ran on CPU.
