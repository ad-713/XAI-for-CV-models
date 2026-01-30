# Chapter Outline & Content Contracts

## 1. Introduction
*   **Objective:** Introduce the "Black Box" problem in Deep CNNs and the imperative for Explainable AI (XAI) in high-stakes domains.
*   **Logical Pillars:**
    *   Context: Ubiquity of CNNs (VGG-16) in image classification.
    *   Problem: The trade-off between accuracy, interpretability, and explanation faithfulness.
    *   Motivation: Trust and accountability in medical/autonomous applications; the need to explore, experiment, and compare different XAI methods.
    *   Contribution: A multi-method comparative analysis (Grad-CAM vs. LIME vs. SHAP) to identify the most effective diagnostic tools for model reliability.

## 2. Related Work
*   **Objective:** Review the state-of-the-art in XAI, establishing the precedence and differences between gradient-based and perturbation-based methods.
*   **Logical Pillars:**
    *   **Visual Explanations:** Evolution from Saliency Maps to Grad-CAM (Gradient-based).
    *   **Model-Agnostic Methods:** Introduction to LIME (Local surrogate models) and SHAP (Shapley values from game theory).
    *   **Gap Analysis:** The need for comparative studies that evaluate not just visual appeal, but also quantitative faithfulness and computational cost.

## 3. Methodology
*   **Objective:** Define the theoretical framework and implementation details for the model and XAI techniques.
*   **Logical Pillars:**
    *   **The Model:** VGG-16 Architecture adapted for CIFAR-10, incorporating Batch Normalization for stability.
    *   **Grad-CAM:** Mathematical derivation of Neuron Importance Weights ($\alpha$) from the final convolutional layer.
    *   **LIME:** Theory of local linear approximations and super-pixel perturbation.
    *   **SHAP:** KernelSHAP implementation using a representative background dataset (100 samples) to estimate Shapley values.

## 4. Experiments
*   **Objective:** Describe the experimental setup, implementation parameters, and evaluation protocols.
*   **Logical Pillars:**
    *   **Dataset (CIFAR-10):** 
        *   60,000 32x32 images across 10 classes.
        *   Preprocessing: Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]).
        *   Augmentation: Random Crop (padding=4) and Random Horizontal Flip.
    *   **Training Configuration:**
        *   Optimizer: SGD (Learning rate: 0.1, Momentum: 0.9, Weight Decay: 5e-4).
        *   Scheduler: Cosine Annealing LR over 50 epochs.
        *   Batch Size: 128.
    *   **Evaluation Protocol:**
        *   Classification Metrics: Accuracy, Confusion Matrix, F1-Score.
        *   XAI Metrics: 
            *   **Faithfulness:** Deletion Score (AUC of probability drop as pixels are removed).
            *   **Efficiency:** Average Runtime per explanation (seconds).

## 5. Results
*   **Objective:** Present quantitative comparison and qualitative visualization galleries.
*   **Logical Pillars:**
    *   **Model Performance:** Accuracy curves and final classification report.
    *   **Quantitative Comparison:** Bar charts comparing Grad-CAM, LIME, and SHAP across Runtime and Deletion Scores.
    *   **Visual Comparative Gallery:** Side-by-side heatmaps for:
        *   Successful classification (discriminative feature focus).
        *   Classification failures (revealing spurious correlations or texture bias).

## 6. Discussion
*   **Objective:** Critically analyze the findings and the trade-offs between XAI methods.
*   **Logical Pillars:**
    *   **Speed vs. Accuracy:** Discussing why Grad-CAM is suitable for real-time needs while SHAP/LIME are better for offline diagnostics.
    *   **Interpreting Failures:** Analyzing specific instances where methods disagreed on the "important" features.
    *   **Practical Implications:** Recommending a workflow for deploying XAI in "Human-in-the-loop" systems based on method performance.

## 7. Conclusion
*   **Objective:** Summarize the study and suggest future research avenues.
*   **Logical Pillars:**
    *   Recap: Comparison demonstrates that no single XAI method is universal; choice depends on the specific domain constraints.
    *   Key Takeaway: Quantitative metrics (Deletion Score) are essential to validate qualitative visualizations.
    *   Future Work: Extension to Vision Transformers (ViT) and automated bias mitigation.
