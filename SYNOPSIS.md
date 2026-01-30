# Project Synopsis: Unveiling the Black Box

## Title
**A Comparative Analysis of Feature-Level Interpretability: Evaluating Grad-CAM, LIME, and SHAP on VGG-16**

## Motivation
This project implements the VGG-16 architecture, a high-performing model on the CIFAR-10 dataset, to explore, experiment with, and compare different explainability methods. As explainability is key in computer vision for sensitive or high-stakes domains, understanding the strengths and weaknesses of various XAI (Explainable AI) techniques is essential for building trust and accountability in deep learning systems.

## Problem Statement
Deep Convolutional Neural Networks (CNNs) demonstrate state-of-the-art performance in image classification but operate as opaque "black boxes." While methods exist to visualize their decision-making process, there is a lack of clear guidance on which methods provide the most faithful and efficient explanations. This ambiguity creates barriers to adoption in domains like medical imaging and autonomous systems.

## Research Question
How do gradient-based (Grad-CAM) and perturbation-based (LIME, SHAP) visualization techniques compare in terms of explanation faithfulness, computational efficiency, and their ability to diagnose classification errors in deep CNNs?

## Hypothesis
We hypothesize that while Grad-CAM offers superior computational efficiency, perturbation-based methods like SHAP and LIME will provide more fine-grained and "faithful" (higher deletion scores) explanations, revealing that many classification errors stem from the model's reliance on spurious background features.

## Scope & Methodology
The project focuses on the VGG-16 architecture trained on the CIFAR-10 benchmark dataset. The methodology involves:
1.  **Baseline Training:** Establishing a competent VGG-16 classifier (with Batch Normalization) on CIFAR-10.
2.  **XAI Integration:** Implementing and configuring three distinct explainability methods:
    *   **Grad-CAM:** Gradient-weighted Class Activation Mapping (last convolutional layer).
    *   **LIME:** Local Interpretable Model-agnostic Explanations (perturbation-based).
    *   **SHAP:** SHapley Additive exPlanations (game-theory based).
3.  **Comparative Analysis:** Systematically evaluating these methods across two dimensions:
    *   **Quantitative:** Measuring Runtime (latency) and Deletion Scores (faithfulness/AUC).
    *   **Qualitative:** Visual comparison of heatmaps for both correct and incorrect predictions to identify failure modes.

## Key Deliverables
1.  A fully trained VGG-16 model with performance metrics on CIFAR-10.
2.  A comparative dashboard showing visual explanations from Grad-CAM, LIME, and SHAP side-by-side.
3.  A quantitative summary report (CSV/Plots) comparing method performance (Runtime vs. Faithfulness).
4.  A diagnostic analysis of model failure modes revealed through multi-method visualization.
