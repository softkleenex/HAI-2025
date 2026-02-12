# Implementation Plan - External Data Injection

## Phase 1: Kaggle Data Setup
- [ ] Task: Locate Celeb-DF dataset on Kaggle.
    - [ ] Subtask: Verify the dataset URL (e.g., `goalavish/celeb-df-v2` or similar).
    - [ ] Subtask: Update `configs/kaggle_celeb_df.yaml` to include both FF++ and Celeb-DF paths.
- [ ] Task: Update Dataset Logic.
    - [ ] Subtask: Modify `src/data/dataset.py` (if needed) to handle multiple root directories or a list of dataset paths.
    - [ ] Subtask: Ensure labels (0 for Real, 1 for Fake) are correctly assigned for Celeb-DF.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Kaggle Data Setup' (Protocol in workflow.md)

## Phase 2: Training Generalist Model
- [ ] Task: Configure Training.
    - [ ] Subtask: Set experiment name to `convnext_generalist`.
    - [ ] Subtask: Use `ConvNeXt-Base` or `EfficientNet-V2-L`.
    - [ ] Subtask: Apply moderate augmentations (Blur/Compression) to match test set quality.
- [ ] Task: Execute Training on Kaggle.
    - [ ] Subtask: Train for 5-10 epochs on the combined dataset.
    - [ ] Subtask: Download the best checkpoint.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Training Generalist Model' (Protocol in workflow.md)

## Phase 3: Ensemble V16
- [ ] Task: Inference.
    - [ ] Subtask: Run TTA inference with the new model.
    - [ ] Subtask: Save `submission_convnext_generalist.csv`.
- [ ] Task: Ensemble.
    - [ ] Subtask: Combine V10 (Pseudo) + V16 (Generalist).
    - [ ] Subtask: Submit and record results.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Ensemble V16' (Protocol in workflow.md)
