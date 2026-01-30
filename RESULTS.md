# Experimental Results

## Model Performance
**Model:** VGG-16 (CIFAR-10 Adapted)
**Best Validation Accuracy:** 92.79%

## Classification Report

### Per-Class Metrics
The following table summarizes the performance for each of the 10 CIFAR-10 classes.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **airplane** | 0.94 | 0.94 | 0.94 | 1000 |
| **automobile** | 0.96 | 0.97 | 0.97 | 1000 |
| **bird** | 0.92 | 0.92 | 0.92 | 1000 |
| **cat** | 0.86 | 0.82 | 0.84 | 1000 |
| **deer** | 0.93 | 0.94 | 0.93 | 1000 |
| **dog** | 0.85 | 0.89 | 0.87 | 1000 |
| **frog** | 0.96 | 0.95 | 0.95 | 1000 |
| **horse** | 0.95 | 0.95 | 0.95 | 1000 |
| **ship** | 0.96 | 0.96 | 0.96 | 1000 |
| **truck** | 0.96 | 0.95 | 0.96 | 1000 |

### Overall Performance Summary

**Overall Accuracy:** **93.00%**

| Average Type | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Avg** | 0.93 | 0.93 | 0.93 | 10000 |
| **Weighted Avg** | 0.93 | 0.93 | 0.93 | 10000 |

## Metric Definitions
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. (Accuracy of positive predictions)
- **Recall:** The ratio of correctly predicted positive observations to the all observations in actual class. (Sensitivity)
- **F1-Score:** The weighted average of Precision and Recall.
- **Support:** The number of actual occurrences of the class in the specified dataset. In this case, there are 1000 images for each of the 10 classes in the CIFAR-10 test set.

## Explainability Analysis

We compared three distinct explainability methods—**Grad-CAM**, **LIME**, and **SHAP**—to evaluate their effectiveness in interpreting the VGG-16 model's decisions.

### Quantitative Comparison
The methods were evaluated based on **Runtime** (efficiency) and **Faithfulness** (Deletion Score).

| Method | Avg Runtime (s) | Avg Deletion Score (AUC) |
| :--- | :--- | :--- |
| **Grad-CAM** | 0.083s | 0.705 |
| **LIME** | 0.384s | 0.692 |
| **SHAP** | 1.386s | 0.267 |

### Analysis
- **Grad-CAM** is the most efficient method, making it suitable for real-time applications, but it offers lower resolution (coarse heatmaps) and is less faithful to the model's decision process compared to SHAP.
- **LIME** provides interpretable superpixel-based explanations but is computationally more expensive than Grad-CAM. Its faithfulness is comparable to Grad-CAM in this experiment.
- **SHAP** (Gradient Explainer) demonstrates superior **faithfulness**, with a significantly lower Deletion Score (0.267 vs ~0.70). This indicates that the pixels highlighted by SHAP are critically important for the model's prediction. However, it is the most computationally intensive method.

### XAI Metric Definitions
- **Deletion Score (Faithfulness):** Measures the drop in the model's confidence for the target class when the most "important" pixels (as identified by the explanation) are iteratively masked. A **lower** score indicates a more faithful explanation, as removing the highlighted features significantly impacts the model's prediction.
- **Runtime:** The average time taken to generate a single explanation on the test hardware.
