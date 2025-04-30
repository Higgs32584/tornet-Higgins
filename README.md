# 🌀 Tornado Prediction with Wide ResNet Ensembles

This contribution extends the [Tornet benchmark](https://github.com/mit-ll/tornet) with optimized deep learning architectures for tornado detection using polarimetric radar data. The focus is on enhancing performance via model refinement and ensembling, culminating in a lightweight and high-performing prediction pipeline.

---

## 📌 Overview

This project:
- Develops multiple ResNet-style neural network variants optimized for the Tornet dataset.
- Integrates **CoordConv2D** to improve spatial awareness in convolutional layers.
- Uses **Precision-Recall AUC** and **Threat Score** as primary metrics to evaluate model performance, particularly for imbalanced classes.
- Introduces a streamlined **ensemble** of Wide ResNet variants that improves robustness and precision across tornado strength levels.

---

## 🧠 Key Contributions

### ✅ Version 5 – Lightweight Wide ResNet
- ~230k parameters.
- Excellent performance-to-size ratio.
- Pre-activation block design with dropout and early normalization.
- Tuned with Binary Cross-Entropy + Adam + Exponential Decay.

### ✅ Version 6 – Gated Wide ResNet
- ~514k parameters.
- Dynamically blends shallow vs. deep inference paths for "easy" and "hard" tornado cases.
- Introduces learned gating for adaptive computation.

### 🔀 Ensemble (v5 + v6)
- Simple average ensemble of model outputs.
- Achieves a **7-point PR-AUC increase** over the baseline.
- Provides robustness across EF0 to EF2+ tornadoes, reducing false alarms while preserving sensitivity.

---

## 📊 Evaluation Metrics

| Metric         | Description                                                 |
|----------------|-------------------------------------------------------------|
| **PR-AUC**     | Threshold-independent evaluation for class imbalance        |
| **F1 Score**   | Precision-recall tradeoff at a specific threshold           |
| **Threat Score (CSI)** | Prioritizes correctness in severe event prediction      |

Ensemble models consistently outperform the [baseline](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1) on all major metrics.

---

## 📁 File Structure

- `scripts/tornado_detection/`
  - `train_wide_resnet.py` – Training logic for WRN variants
  - `train_gated_routing.py` – Model v6 with learned gate logic
  - `test_tornado_keras_batch.py` – Batch evaluation and ensemble runner
- `models/`
  - Saved `.keras` models for versioned checkpoints
- `visualizations/`
  - Plots of AUCPR, precision-recall tradeoffs, and architecture comparisons

---

## 📷 Sample Visualizations

![AUCPR Over Epochs](docs/images/aucpr_over_epochs.png)
*Figure: PR-AUC convergence of different model versions.*

![Model Architecture Comparison](docs/images/resnet_block_comparison.png)
*Figure: Comparison of baseline vs. WRN (v5/v6) building blocks.*

---

## 🔬 Dependencies

- Python 3.10+
- TensorFlow 2.15+
- Keras
- TFDS (for Tornet dataset)
- `matplotlib`, `scikit-learn`, `tqdm`, etc.

Install dependencies:
```bash
pip install -r requirements.txt


### Disclosure
```
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
```
