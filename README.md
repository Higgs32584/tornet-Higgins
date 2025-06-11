# 🌀 Update 6-11-2025

This contribution extends the [Tornet benchmark](https://github.com/mit-ll/tornet) with optimized deep learning architectures for tornado detection using polarimetric radar data. The focus is on enhancing performance via model refinement and ensembling, culminating in a lightweight and high-performing prediction pipeline that utilizes wide_resnet with a gated mechanism, as well as an 85 percent reduction in memory

---

## 📌 Overview (Update 6-11-2025)

This update is a bit tricky because the validation metrics and test metrics do not align. You can view the validation scores on Hugging Face.

Here’s what happened: normally, I take the best-performing cross-validation model from each fold, ensemble them, and proceed from there. In this case, the cross-validation model from fold 1 performed best during cross-validation but ended up performing worse than version 8 on the test set.

As a result, I decided to create a new model:

Version 8: the original ensemble script.

Version 9: an ensemble of the absolute best-performing model from each fold.

I also observed that gains across folds were inconsistent — a model that was marginally better in one fold could be slightly worse in another. Because of this, I started saving the absolute best-performing model and script for each fold and used that to create version 9 (which is slightly worse overall than version 8).

I saved the corresponding scripts for each best fold. However, it is now clear that the cross-validation results are starting to diverge from the test set performance.

Given this divergence, I suspect we might be approaching a performance ceiling. It’s hard to say for certain, but temporal modeling might squeeze out a few more points. However, with gains now being inconsistent, it’s difficult to predict. More data could potentially help.







Overall, a lot of the gain came from two main factors

## Dilation Rates
Dilation rates proved highly effective in allowing the model to gain a broader context on the actual image. in simpler terms, dilation rates drag the filter over the model to get the broader context of the model. But we keep the same 3x3 kernel, so we aren't actually doing any more computation. the dilation rate is n-1 pixels between each of the convolutional pixels, so if it is two, there is one pixel between each 3x3 kernel.

![image](https://github.com/user-attachments/assets/9968e6f3-f980-49b9-903b-208e6ec26a74)


## Spatial Dropout 2D
Spatial Dropout also proved effective, especially across fold 1. Granted, some folds still respond better with regular dropout. In simpler terms, spatial dropout drops entire channels, so we never botch out random pixels in a feature map.
![image](https://github.com/user-attachments/assets/1c4c9c46-42bf-4e6e-8651-45f34c7a9738)

# Possible Future Improvements

-	Experiment with **different dilation rates**, or even possibly **different kernel sizes** we can also look at different stride lengths as well.
-	Experiment with Multi-dilated Wide Residual Networks, as this may prove to be more effective than regular convolution.
-	Experiment with hard or soft gating for the actual code for the multi-dilation convolutional code, determine if **SelectAttentionBranch** or the other version is better for prediction
-	Experiment with the tradeoffs between SpatialDropout2D and Dropout2D
-	Further Parameter Tuning could potentially help
-	Determine why superior cross validation on training data does not yield a higher score on the test data
-	Experiment with temporal Modeling to determine what the best course of action is in the future. That is likely where future gains will come from
-	With the introduction of SpatialDropout2D, the model can go far deeper without losing linear separation. Determine if there is any benefit derived from Going deeper
-	Overall, each fold seems to excel largely in vastly different types of scripts at this point, determine why that might possibly be the case.
-	Experiment with **BinaryFocalCrosssEntropy** vs **BinaryCrossEntropy** as this might hold the key to a more linearly separable dataset.
-	Experiment with more advanced neural network architectures.

This will probably be my last update for a long time, thank you all.

---

---

## 📊 Evaluation Metrics

### 📈 Model Comparison Table
| Model Name        | Threshold | Model parameters | AUC   | AUCPR  | BinaryAccuracy | TruePositives | FalsePositives | TrueNegatives | FalseNegatives | Precision | Recall | FalseAlarmRate | F1    | ThreatScore |
|-------------------|-----------|------------------|-------|--------|----------------|---------------|----------------|---------------|----------------|-----------|--------|----------------|-------|-------------|
| baseline          | 0.0101    | 4665409          | 0.8742 | 0.5349 | 0.9456         | 915           | 635            | 28841         | 1076           | 0.5903    | 0.4596 | 0.4097         | 0.5168 | 0.3484      |
| WRN               | 0.4444    | 229759           | 0.882  | 0.5498 | 0.9456         | 973           | 693            | 28783         | 1018           | 0.5840    | 0.4887 | 0.416          | 0.5321 | 0.3625      |
| gated             | 0.5353    | 514905           | 0.8862 | 0.5574 | 0.9436         | 976           | 761            | 28715         | 1015           | 0.5619    | 0.4902 | 0.4381         | 0.5236 | 0.3547      |
| WRN+Gated         | 0.4898    | 744664           | 0.8928 | 0.5705 | 0.9474         | 975           | 640            | 28836         | 1016           | 0.6037    | 0.4897 | 0.3963         | 0.5408 | 0.3706      |
| PReLU_textbook    | 0.4242    | 102524           | 0.8976 | 0.5853 | 0.9481         | 1032          | 673            | 28803         | 959            | 0.6053    | 0.5183 | 0.3947         | 0.5584 | 0.3874      |
| old_ensemble      | 0.3899    | 512620            | 0.9047 | 0.5947 | 0.9479         | 1020          | 669            | 28807         | 971            | 0.6039    | 0.5123 | 0.3961         | 0.5543 | 0.3835      |
| ensemble_V8TH     | 0.3999    | 1050714          | 0.911  | 0.6179 | 0.9497         | **1068**      | 659            | 28817         | **923**        | 0.6184    | **0.5364** | 0.3816         | **0.5745** | **0.403**       |
| ensemble_v9       | 0.46      | 1379898          | 0.9115 | 0.6143 | 0.9492         | 1038          | 644            | 28832         | 953            | 0.6171    | 0.5213 | 0.3829         | 0.5652 | 0.3939      |
| ensemble_v8+v9    | 0.4353    | **2430612**      | **0.9128** | **0.6195** | **0.9504**    | 1050          | **621**        | **28855**     | 941            | **0.6284** | 0.5274 | **0.3716**     | 0.5735 | 0.402       |




### 📈 Cross-Validation AUCPR

| Name                    | seed   | Params | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | Average |
|-------------------------|--------|--------|--------|--------|--------|--------|--------|---------|
| baseline                |        | 4.5M   | 0.5121 | 0.4405 | 0.4647 | 0.4544 | 0.4451 | 0.4634  |
| WRN                     |        | 230k   | 0.6184 | 0.5546 | 0.5986 | 0.6152 | 0.6072 | 0.5988  |
| Gated                   |        | 600k   | 0.6401 | 0.5516 | 0.5247 | 0.5878 | 0.5716 | 0.5752  |
| PReLU                   |        | 102k   | 0.6314 | 0.6210 | 0.6006 | 0.6266 | 0.6132 | 0.6186  |
| ensemble_v8             |        |        | 0.6338 | **0.6359** | 0.6141 | 0.6342 | **0.6211** | 0.62782 |
| fold 1 script cross val |        | ~300k  | **0.6627** | 0.6312 | **0.6171** | **0.6488** | 0.6187 | **0.6357** |





### 📈 Cross Valdiation AUC (taken on best run of AUCPR)

### 📈 Cross-Validation AUC (Best AUCPR Runs)

| Name     | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | Average |
|----------|--------|--------|--------|--------|--------|---------|
| baseline | 0.8950 | 0.8744 | 0.8720 | 0.8602 | 0.8708 | 0.8745  |
| wide     | 0.8920 | 0.8697 | 0.8824 | 0.8914 | 0.9077 | 0.8886  |
| gated    | 0.8893 | 0.8749 | 0.8694 | 0.8877 | 0.8927 | 0.8828  |
| PReLU_textbook_f5    | **0.9017** | **0.8940** | **0.8882** | **0.8954** | **0.9128** | **0.8984**  |



**Key Points:**




Ensemble models consistently outperform the [baseline](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1) on all major metrics.

---

## 📁 File Structure

### `scripts/tornado_detection/`
- `foldxx.py` the respective scripts for the best scoring folds overall for each model

### `visualizations/`
- Plots showing AUCPR performance, precision-recall tradeoffs, and model architecture comparisons
![image](https://github.com/user-attachments/assets/b40964ff-37fa-4262-9c85-acb2f172fe1e)



## If you have any questions feel free to open an issue. Thank you

## 🧪 Ensemble Evaluation Usage
Put all .keras in the same file
```bash
python scripts/tornado_detection/test_tornado_keras_ensemble.py \
  --model_dir MODEL_DIR_PATH \
  --threshold THRESHOLD
```
---
## Downloading the Data and set up is the same as [Tornet benchmark](https://github.com/mit-ll/tornet)


## Citation
```
If you use this repository, its models, or training scripts in any academic, commercial, or public work,**you must cite the following**:
@misc{higgins2025tornet,
  author = {Michael Higgins},
  title = {Improved Tornado Detection with Wide ResNet on TorNet},
  year = {2025},
  url = {https://github.com/Higgs32584/tornet-Higgins}
}
```
## DISTRIBUTION STATEMENT 

### Disclosure
```
MIT License

Copyright (c) 2025 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell    
copies of the Software, and to permit persons to whom the Software is        
furnished to do so, subject to the following conditions:                     

The above copyright notice and this permission notice shall be included in   
all copies or substantial portions of the Software.                          

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN    
THE SOFTWARE.
```
