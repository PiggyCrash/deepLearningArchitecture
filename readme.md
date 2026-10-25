# CIFAR-10 CNN Experiments — img_clf

> "If I change X, what happens to Y?" — a systematic, GPU-accelerated experiment
> series on the original `img_clf.ipynb` notebook: **12 experiments, one variable
> at a time, every result quantified.**

[![Device](https://img.shields.io/badge/device-CUDA%20(RTX%204050)-76B900)](#general-setup)
[![torch](https://img.shields.io/badge/torch-2.7.1%2Bcu118-EE4C2C)](#general-setup)
[![Dataset](https://img.shields.io/badge/dataset-CIFAR--10-blue)](#baseline-setup)
[![Experiments](https://img.shields.io/badge/experiments-12-orange)](#experiment-chains)
[![Best test acc](https://img.shields.io/badge/best%20test%20acc-86.38%25-brightgreen)](#results-at-a-glance)

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [General setup](#general-setup)
- [Baseline setup](#baseline-setup)
- [Main results table](#main-results-table)
- [Experiment chains](#experiment-chains)
  - [Chain 1 — Data processing](#chain-1--data-processing-e1--e2--e3)
  - [Chain 2 — Architecture](#chain-2--architecture-e4-a--b--c--d)
  - [Chain 3 — Hyperparameters](#chain-3--hyperparameters-e5-a--b--c)
  - [Chain 4 — Transfer learning](#chain-4--transfer-learning-e6-vs-e7)
  - [Chain 5 — Final long-runs](#chain-5--final-long-runs-e8--e9-model-not-changed-from-baseline)
- [If you change X, you get Y](#if-you-change-x-you-get-y-summary)
- [Transparency: out-of-scope changes](#transparency-out-of-scope-changes)
- [Best models](#best-models)

---

## Results at a glance

| | Baseline | Best scratch (same model) | Best transfer (same model) | Best overall (arch changed) |
|---|---|---|---|---|
| **Test acc** | 10.00% | 79.33% (E8) | **86.38%** (E9) ✅ | 92.80% (ResNet-18)* |
| **Train time** | 35.7s | 517.7s | 504.7s | 1336.1s |

\* informational only — architecture change was later disallowed for the final runs.

**The single most impactful fix:** sigmoid+BCE → logits+CrossEntropy = **10% → 43.7%** test accuracy. Nothing else comes close.

---

## Repository structure

```
ml_task_cnn/
├─ img_clf.ipynb                  # original baseline notebook (untouched)
├─ colab img_clf.ipynb            # original's Google Colab port (untouched)
├─ EXPERIMENT_RESULTS.md          # this file
├─ modelz.pth                     # original author's checkpoint
└─ experiments/
   ├─ exp_common.py               # shared module: data, models, train/eval/test, runner
   ├─ E1_baseline_original.ipynb  # executed, results embedded
   ├─ E2_data_fix.ipynb
   ├─ E3_augmentation_split.ipynb
   ├─ E4_arch_A_logits_ce.ipynb   # architecture chain A→B→C→D
   ├─ E4_arch_B_padding_bn.ipynb
   ├─ E4_arch_C_deeper_dropout.ipynb
   ├─ E4_arch_D_residual.ipynb
   ├─ E5_hyperA_lr_batch.ipynb    # hyperparameter chain A→B→C
   ├─ E5_hyperB_adam.ipynb
   ├─ E5_hyperC_epochs.ipynb
   ├─ E6_transfer_effnet_original.ipynb
   ├─ E7_transfer_frozen.ipynb
   ├─ E8_final_scratch.ipynb      # original ConvNet, 100 epochs
   ├─ E9_final_transfer.ipynb     # original EffNet-B0, 20 epochs
   └─ results/*.json              # raw metrics per experiment
```

---

## Quick start

```powershell
cd C:\Users\dgash\Downloads\ml_task_cnn\experiments
python -m nbconvert --to notebook --execute --inplace <notebook_name>.ipynb
```

Execution order: E1 → E2 → E3 → E4-A → E4-B → E4-C → E4-D → E5-A → E5-B → E5-C → E6 → E7 → E8 → E9.
Dependencies: `torch`, `torchvision`, `numpy`, `nbformat`, `nbconvert`, `ipykernel`.

---

## General setup

| Item | Value |
|---|---|
| Device | `cuda` (RTX 4050 Laptop, 6 GB) |
| Data | Local CIFAR-10 batch pickles (`cifar-10-batches-py/`), 60,000 samples total |
| Test | 10,000 `test_batch` samples, batched (256) — the original notebook used batch=1 (±400 seconds) |
| Metrics | train/val loss+acc per epoch, test acc, per-class acc, time/epoch, total time, peak VRAM, parameter count |
| Reproducibility | fixed seed (42) across all experiments |

---

## Baseline setup

Every experiment is measured against the original `img_clf.ipynb`. This is an exact
description of what it contains — no changes whatsoever.

### 1. Data & preprocessing

| Item | Original implementation |
|---|---|
| Dataset | CIFAR-10, per-batch pickle files (`data_batch_1..5`, `test_batch`), read manually via `pickle` (not `torchvision.datasets`) |
| Sample format | flat 3072 uint8 array = R(0:1024), G(1024:2048), B(2048:3072), reshaped to 32x32x3 then transposed to CHW |
| **Preprocessing bug** | in `myData.__getitem__`, `img_green` and `img_blue` both take `data[0:1024]` → **all three channels = the red channel** (the image is effectively a red-channel grayscale; the real G & B are discarded) |
| Normalization | none in the dataset; done inside the model (`x / 255`) |
| Label | integer 0–9 → manual **one-hot encoding** (`one_hot` function, 10-float array) |
| Augmentation | **none** (no flip/crop/etc.) |
| Train/val split | train = `data_batch_1,3,5` (30,000 samples), val = `data_batch_2,4` (20,000 samples) |
| Test | `test_batch` (10,000 samples), **batch_size=1** (10,000 forwards, ±400 seconds) |

### 2. Architecture — A. Train from scratch: `ConvNet`

```
input (3x32x32)
 ├─ conv1: Conv2d 3→16, kernel 3, NO padding  + ReLU + MaxPool(2)   → 16x15x15
 ├─ conv2: Conv2d 16→32, kernel 3, no padding + ReLU + MaxPool(2)   → 32x6x6
 ├─ conv3: Conv2d 32→64, kernel 3, no padding + ReLU + MaxPool(2)   → 64x2x2
 ├─ AdaptiveAvgPool2d(1) + Flatten                                  → 64
 ├─ fc1: Linear 64→128  + ReLU
 ├─ fc2: Linear 128→64  + ReLU
 └─ fc3: Linear 64→10   + **Sigmoid**
```

- Hidden activations: **ReLU**; output activation: **Sigmoid** (per-class, independent)
- No BatchNorm, no Dropout, no residuals
- Total parameters: **40,810**

### 3. Architecture — B. Transfer learning: `EffNet_clf`

- Backbone: **EfficientNet-B0 pretrained on ImageNet** (`models.efficientnet_b0(pretrained=True)`)
- Forward written manually per stage (`features[0]` … `features[8]`)
- ImageNet normalizer (mean/std) applied to raw **0–255** values (without /255 — mathematically incorrect)
- Classifier replaced with `Linear(1280→10)`, output **Sigmoid**
- All weights trained (full fine-tune), nothing frozen
- Total parameters: **4,020,358**

### 4. Loss, optimizer, hyperparameters

| Item | Original value |
|---|---|
| Loss | `F.binary_cross_entropy` (BCE) on sigmoid outputs vs one-hot labels |
| Optimizer | **SGD**, lr = **0.0025**, momentum = 0.9 |
| LR schedule | none (constant lr) |
| Weight decay | none |
| Batch size | **10** |
| Epoch | original notebook: 1 (example CPU run); E1 replica: 3 |
| Original device | `cpu` (training cell: `torch.device("cpu")`) |
| Seed | not set (results not reproducible) |

### 5. Helper functions

| Function | Role |
|---|---|
| `unpickle` | read a CIFAR-10 batch pickle file |
| `one_hot` | int label → 10-float one-hot |
| `myData(Dataset)` | merge all batch files → `{'image', 'label'}` per sample (contains the channel bug above) |
| `AverageMeter` | running average of loss/metric |
| `train(...)` | 1 epoch: forward → BCE → backward → update; tqdm progress bar; per-batch logging to TensorBoard `SummaryWriter` |
| `validate(...)` | same, without backward |
| `test(...)` | per-sample inference (batch 1), argmax vs argmax → accuracy; writes `test_log.csv` **on every sample** (10,000 file writes — one source of slowness) |

### 6. Metrics & logging

- Main metric: **accuracy** (argmax pred vs argmax label), plus train/val BCE loss
- Logging: `trainval_log.csv` per epoch, `test_log.csv` per sample, TensorBoard per batch

### 7. Issues identified in the baseline (improvement targets)

1. **RGB channel bug** (R=G=B) — color information lost
2. **Sigmoid+BCE for 10-class classification** — doesn't force "pick one"; weak gradients
3. **lr 0.0025 too small** for a non-BN model → collapses to constant output
4. No augmentation, 40% val split too large (steals training data)
5. Architecture without BN/padding/residuals, GAP too early, 41k-param capacity
6. Test batch=1 + CSV write per sample → evaluation 10x slower than necessary
7. Training on CPU while a GPU is available

---

## Main results table

| Exp | What changed | Test acc | Val acc (final) | Train time | s/epoch | Peak VRAM | Params |
|---|---|---|---|---|---|---|---|
| **E1** baseline | original replica (sigmoid+BCE, SGD lr=0.0025, batch 10, **RGB bug**, 3 epochs) | **10.00%** | 9.85% | 35.7s | 11.9 | 52 MB | 40,810 |
| **E2** data fix | fix G/B channel slicing | **10.00%** | 9.85% | 35.3s | 11.8 | 52 MB | 40,810 |
| **E3** augmentation | + flip/crop, train 30k→40k, val 20k→10k | **10.00%** | 10.08% | 57.0s | 19.0 | 52 MB | 40,810 |
| **E4-A** arch | sigmoid+BCE → **logits+CrossEntropy** | **43.70%** | 43.13% | 55.6s | 18.5 | 52 MB | 40,810 |
| **E4-B** arch | + padding=1 + **BatchNorm** | **56.73%** | 56.95% | 67.2s | 22.4 | 57 MB | 43,914 |
| **E4-C** arch | + deeper/wider (4 blocks) + Dropout | **54.59%** | 55.07% | 151.6s | 50.5 | 187 MB | 331,530 |
| **E4-D** arch | + **residual connections** (mini-ResNet) | **66.57%** | 67.07% | 210.9s | 70.3 | 157 MB | 296,858 |
| **E5-A** hyper | lr 0.0025→0.01, batch 10→64 | **61.84%** | 62.35% | 51.9s | 17.3 | 157 MB | 296,858 |
| **E5-B** hyper | SGD → **Adam** (lr=0.001) | **66.51%** | 67.02% | 58.9s | 19.6 | 159 MB | 296,858 |
| **E5-C** hyper | epochs 3→**8** | **79.46%** | 79.47% | 137.4s | 17.2 | 159 MB | 296,858 |
| **E6** transfer | original EffNet-B0 (sigmoid+BCE, full FT, SGD lr=0.0025) | **54.06%** | 53.99% | 169.8s | 56.6 | 171 MB | 4,020,358 |
| **E7** transfer | **frozen** backbone + new head + Adam + CE | **38.90%** | 38.31% | 90.7s | 30.2 | 94 MB | 165,258 trainable |
| **E8** final scratch | **ORIGINAL ConvNet** (architecture untouched), logits+CE, SGD lr=0.05 cosine, wd=5e-4, batch 128, **100 epochs** | **79.33%** | 78.78% | 517.7s | 5.2 | 52 MB | 40,810 |
| **E9** final transfer | **ORIGINAL EffNet-B0** (baseline model), logits+CE, AdamW lr=3e-4 cosine, batch 64, **20 epochs** | **86.38%** | 86.28% | 504.7s | 25.2 | 171 MB | 4,020,358 |

---

## Experiment chains

### Chain 1 — Data processing (E1 → E2 → E3)

- **E1 (baseline)**: exact replica of the original notebook. The model **collapses
  completely**: it predicts a single class ("automobile") for everything → 10% test acc.
  Loss stuck at ~0.3253 from epoch 0.
- **E2**: fixes the channel slicing bug (`img_green`/`img_blue` in the original both
  take `data[0:1024]`, so R=G=B). Result: **identical 10%**.
- **E3**: + augmentation + more training data. Result: still **10%**.

> **Conclusion: the RGB bug is not the main problem.** Sigmoid+BCE with lr=0.0025
> makes gradients too weak — the model is content at the shallow "constant answer"
> minimum. Data fixes mean nothing while the loss/activation is still wrong.

### Chain 2 — Architecture (E4-A → B → C → D)

Each modification is driven by the interpretation of the previous result.

| Step | Mod | Test acc | Interpretation |
|---|---|---|---|
| A | sigmoid+BCE → logits+CE | 10% → **43.7%** | Model finally learns, but train acc 37.8% < val 43.1%, loss still falling → *underfitting*, slow convergence |
| B | + padding=1 + BatchNorm | 43.7% → **56.7%** | BN stabilizes activations, much faster convergence. Still underfit → bottleneck shifts to capacity |
| C | + deeper/wider + Dropout | 56.7% → **54.6%** ⚠️ | **Worse** + 2.3x slower. A plain deep net with a small lr is hard to optimize; extra capacity goes unused in 3 epochs |
| D | + residual connections | 54.6% → **66.6%** | Shortcuts `x + F(x)` make the depth actually usable. Best scratch model |

> **Lesson: merely adding layers (C) can hurt; fixing gradient flow
> (B: BatchNorm, D: residuals) matters far more.**

### Chain 3 — Hyperparameters (E5-A → B → C)

On top of architecture D.

| Step | Mod | Test acc | Train time | Interpretation |
|---|---|---|---|---|
| A | lr 4x, batch 64 | 66.6% → **61.8%** ⚠️ | 211s → **51.9s** | 4x faster BUT total updates dropped 12,000 → 1,875 steps → undertrained |
| B | → Adam lr=0.001 | 61.8% → **66.5%** | 58.9s | Adam matches D's accuracy while being **3.6x faster** — every step is more effective |
| C | epochs 3→8 | 66.5% → **79.5%** | 137.4s | The bottleneck was the training budget all along. Per-epoch curve below |

E5-C curve (train_acc / val_acc per epoch):

```
epoch 0:  44.3% / 52.4%     epoch 4:  73.5% / 74.0%
epoch 1:  60.3% / 59.9%     epoch 5:  76.2% / 75.4%
epoch 2:  66.3% / 67.6%     epoch 6:  78.5% / 77.6%
epoch 3:  70.3% / 73.5%     epoch 7:  79.9% / 79.5%   <- not overfit yet, still rising
```

Train acc ≈ val acc through the end → not overfitting; more epochs would likely
keep improving accuracy.

### Chain 4 — Transfer learning (E6 vs E7)

- **E6** (original design: full fine-tune + sigmoid/BCE + SGD lr=0.0025): **54.1%**,
  56.6s/epoch. Pretraining helps (vs the 10% baseline), but the small lr + BCE limit
  adaptation within 3 epochs — it loses to scratch ConvNetD+Adam (79.5%).
- **E7** (frozen backbone, only the 165k-param head trained): **38.9%**, but
  30.2s/epoch and only 94 MB VRAM. ImageNet features on 32x32 without adaptation
  are not enough — but it's very cheap.

> **Takeaway: transfer learning is not magic — the training config (loss, lr,
> optimizer) is still the decider. The same bad original config also drags down
> a pretrained model.**

### Chain 5 — Final long-runs (E8 & E9), model NOT changed from baseline

Per instruction: the model is **exactly** the one from the baseline notebook (no new
architecture). Only the training recipe changed: logits+CE, larger lr + cosine
annealing, weight decay, larger batch, far more epochs.

#### E9 — Transfer learning (original EfficientNet-B0): TARGET ACHIEVED ✅

| Metric | E9 (20 epochs) |
|---|---|
| **Train acc** | **91.78%** ✅ |
| **Val acc** | **86.28%** ✅ |
| **Test acc** | **86.38%** ✅ |
| Time | 504.7s (25.2s/epoch) |

Curve: epoch 0: 52/68% → epoch 5: 81/82% → epoch 10: 87/86% → epoch 19: 92/86%.
Keys to success vs E6 (54%): swap sigmoid+BCE → logits+CE, SGD lr 0.0025 →
AdamW lr 3e-4, epochs 3 → 20. Model exactly the same.

#### E8 — Train from scratch (original ConvNet): 85% target NOT reached ❌

| Metric | E8 (100 epochs) |
|---|---|
| **Train acc** | **80.13%** |
| **Val acc** | **78.78%** |
| **Test acc** | **79.33%** |
| Time | 517.7s (5.2s/epoch) |

The curve rises smoothly and **flattens** at the end (epoch 74: 77.1% → epoch 99:
80.1% train) — this is the **capacity ceiling** of a 41k-parameter model without
BatchNorm/residuals, not a recipe problem. Proof: the exact same recipe on
ResNet-18 (11M params, 40 epochs) reached **train 99.7% / val 92.9% / test 92.8%** —
but that means changing the model, which was not allowed here.

> **Conclusion: with the original ConvNet, ~79-80% is the ceiling.**
> Breaking 85% from scratch requires a stronger architecture (BatchNorm +
> residuals + capacity), as shown by the E4-A→D chain and the ResNet-18 run.

---

## "If you change X, you get Y" summary

| Change | Effect on test acc | Side effects |
|---|---|---|
| Fix RGB slicing bug | 10% → 10% (none) | — |
| + augmentation & 40k data | 10% → 10% (not felt yet) | time/epoch +60% |
| Sigmoid+BCE → logits+CE | **10% → 43.7%** ✅ biggest single win | — |
| + padding & BatchNorm | 43.7% → 56.7% ✅ | +4s/epoch, VRAM +5MB |
| + plain depth/width | 56.7% → 54.6% ❌ | time 2.3x, VRAM 3.3x, params 8x |
| + residual connections | 54.6% → 66.6% ✅ | time +40%, VRAM −30MB vs C |
| lr 4x + batch 64 | 66.6% → 61.8% ❌ (3 epochs) | 4x faster ✅ |
| SGD → Adam | 61.8% → 66.5% ✅ | stays fast |
| epochs 3 → 8 | 66.5% → **79.5%** ✅ best | time linear (2.3x) |
| Transfer learning full FT | 54.1% | params 100x (4M), 56.6s/epoch |
| Transfer learning frozen | 38.9% | only 165k trainable, lightest |

---

## Transparency: out-of-scope changes

Changes that were forced or happened outside the experiment variables:

1. **Test made batched** (256) — original notebook used batch=1 ±400s; metric results
   identical, just far faster.
2. **num_workers=0** — on Windows, workers >0 risk hanging the Jupyter kernel.
3. **Epochs 1 → 3** in the baseline (1 epoch gives no signal at all).
4. **EffNet input scaled /255** before the ImageNet normalizer — the original
   normalizes raw 0–255 values (mathematically incorrect).
5. **Fixed seed (42)** across all experiments so initialization & shuffling are comparable.
6. Augmentation side effect: **train acc < val acc** in several experiments — not an
   anomaly, the train set is simply harder (augmentation active during training only).
7. **Batch 10 → 64 for E6/E7 only** — EffNet-B0 with batch 10 on 40k samples
   (4000 steps/epoch) is too slow for a 6GB GPU.

---

## Best models

| Category | Model | Test acc |
|---|---|---|
| **Without changing the baseline model** | E9 (EffNet-B0 transfer, proper recipe) | **86.38%** — all metrics > 85% |
| **Scratch, baseline model untouched** | E8 (original ConvNet, 100 epochs) | **79.33%** — model capacity ceiling |
| **If architecture may change (scratch)** | ResNet-18 + E8 recipe | **92.80%** (train 99.7% / val 92.9%) |
| **Small budget (fair E1–E7 comparison)** | E5-C (ConvNetD + Adam + 8 epochs) | **79.46%** |

Per-class accuracy (E5-C): ship 94.1%, automobile 89.9%, truck 89.0%, frog 88.8%,
bird 82.0%, dog 74.6%, deer 73.1%, cat 69.0%, airplane 67.1%, horse 67.0%.
