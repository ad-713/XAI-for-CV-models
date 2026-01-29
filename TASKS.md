# Project Tasks Roadmap

## Phase 1: Research & Synthesis (Researcher Mode)
- [ ] **Literature & Related Work:**
  - [ ] Synthesize *State_of_Art.docx* into a coherent "Related Work" section narrative.
  - [ ] Extract key comparisons between Grad-CAM and LIME (Speed, Stability, Resolution).
  - [ ] Find authoritative citations for VGG-16, Grad-CAM, and the "Black Box" problem.
- [ ] **Methodology Definition:**
  - [ ] Document the mathematical formulation of Grad-CAM (Gradient calculation, Global Average Pooling).
  - [ ] Define the specific VGG-16 architecture details (layers, parameters) to be used.
  - [ ] Select and document the dataset specifications (CIFAR-10 vs ImageNet subset).
- [ ] **Experimental Protocol:**
  - [ ] Define specific hyperparameters for training (Learning rate, Optimizer, Batch size).
  - [ ] Define the exact metrics for Quantitative Evaluation (Accuracy, Precision, Recall, F1).

## Phase 2: Content Drafting (Writer Mode)
- [ ] **Draft Introduction:**
  - [ ] Write the hook (Black Box problem) and Motivation (Trust).
  - [ ] State Objectives and Contributions clearly.
- [ ] **Draft Related Work:**
  - [ ] Write the narrative survey of XAI methods.
  - [ ] Write the specific comparison of Grad-CAM vs LIME.
- [ ] **Draft Methodology:**
  - [ ] Write the "Model Architecture" subsection (VGG-16).
  - [ ] Write the "Grad-CAM Technique" subsection (Math & Logic).
- [ ] **Draft Experiments:**
  - [ ] Detail the Dataset and Preprocessing steps.
  - [ ] Detail the Training Setup.
- [ ] **Draft Results:**
  - [ ] Create placeholder tables for Quantitative Results.
  - [ ] Describe the "Correct Prediction" visualizations (What should we see?).
  - [ ] Describe the "Failure Case" visualizations (The diagnostic analysis).
- [ ] **Draft Discussion & Conclusion:**
  - [ ] Interpret the failure modes.
  - [ ] Summarize findings and future work.

## Phase 3: Typesetting & Formatting (Typesetter Mode)
- [ ] **LaTeX Setup:**
  - [ ] Initialize `main.tex` with a standard double-column academic class (e.g., `IEEEtran` or `acmart`).
  - [ ] Configure `biblatex` for references.
- [ ] **Document Composition:**
  - [ ] Convert Markdown drafts into LaTeX chapters/sections.
  - [ ] Insert placeholder figures and tables.
- [ ] **Final Polish:**
  - [ ] Verify reference consistency.
  - [ ] Check formatting constraints (page limits, double-column layout).
