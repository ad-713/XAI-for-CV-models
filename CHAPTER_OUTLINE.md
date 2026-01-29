# Chapter Outline & Content Contracts

## 1. Introduction
*   **Objective:** Introduce the "Black Box" problem in Deep CNNs and the imperative for Explainable AI (XAI) in high-stakes domains.
*   **Logical Pillars:**
    *   Context: ubiquity of CNNs (VGG-16) in image classification.
    *   Problem: The trade-off between accuracy and interpretability.
    *   Motivation: Trust and accountability in medical/autonomous applications.
    *   Contribution: A feature-level diagnostic analysis using Grad-CAM to validate model reliability.
*   **Research Needs:** Statistics on CNN adoption; citations on the "right to explanation" (GDPR or ethical AI frameworks).

## 2. Related Work
*   **Objective:** Review the state-of-the-art in XAI, establishing the precedence of Grad-CAM.
*   **Logical Pillars:**
    *   **Visual Explanations:** Evolution from Saliency Maps to CAM and Grad-CAM.
    *   **Model-Agnostic vs. Specific:** Contrast Grad-CAM (Gradient-based) with LIME (Perturbation-based) using insights from *State_of_Art.docx*.
    *   **Gap Analysis:** Limitations of current methods (computational cost of LIME, coarseness of CAM) and the need for diagnostic application.
*   **Research Needs:** Synthesize comparisons from *State_of_Art.docx*; Comparison table of XAI methods (Speed vs. Resolution).

## 3. Methodology
*   **Objective:** Define the theoretical framework and algorithms used.
*   **Logical Pillars:**
    *   **The Model:** VGG-16 Architecture overview (Convolutional blocks, Fully Connected layers).
    *   **The Technique:** Mathematical derivation of Grad-CAM.
        *   Calculation of Neuron Importance Weights ($\alpha$).
        *   Weighted combination and ReLU activation.
    *   **Integration:** How Grad-CAM is hooked into the final convolutional layer of VGG-16.
*   **Research Needs:** Formal equations for Grad-CAM; Architecture diagram of VGG-16 showing the tap point.

## 4. Experiments
*   **Objective:** Describe the experimental setup and implementation details to ensure reproducibility.
*   **Logical Pillars:**
    *   **Dataset:** Description of CIFAR-10/ImageNet subset (Classes, Train/Test split, Preprocessing).
    *   **Training Configuration:** Hyperparameters (Learning rate, Batch size, Optimizer, Epochs).
    *   **Evaluation Protocol:**
        *   Metric selection (Accuracy, F1).
        *   Qualitative assessment criteria for heatmaps (Localization accuracy).
*   **Research Needs:** Exact training parameters; Hardware specifications (GPU used); Software stack (PyTorch/TensorFlow versions).

## 5. Results
*   **Objective:** Present quantitative performance and qualitative visualization galleries.
*   **Logical Pillars:**
    *   **Model Performance:** Tabular presentation of classification metrics.
    *   **Visual Explanations (Correct Predictions):** Heatmaps showing focus on discriminative features (e.g., snout of a dog).
    *   **Visual Explanations (Failure Cases):** Heatmaps revealing spurious correlations (e.g., focus on water instead of the ship).
*   **Research Needs:** Generated plots (Loss/Accuracy curves); Selected Grad-CAM overlay images.

## 6. Discussion
*   **Objective:** Critically analyze the findings and their implications for model trust.
*   **Logical Pillars:**
    *   **Interpretation of Failures:** Discussing *why* the model failed based on visual evidence (e.g., background bias).
    *   **Grad-CAM Assessment:** Evaluating the tool itself – did it provide actionable insight?
    *   **Practical Implications:** How this workflow applies to real-world deployment (e.g., "Human-in-the-loop" systems).
*   **Research Needs:** Connection back to the "Trust" motivation in Intro.

## 7. Conclusion
*   **Objective:** Summarize the study and suggest future research avenues.
*   **Logical Pillars:**
    *   Recap: Grad-CAM successfully illuminates VGG-16 decision paths.
    *   Key Takeaway: Visualization is essential for debugging, not just validation.
    *   Future Work: Applying to Transformers (ViT); Automated bias detection.
*   **Research Needs:** None.
