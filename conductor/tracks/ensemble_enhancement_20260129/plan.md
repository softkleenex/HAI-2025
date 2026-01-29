# Implementation Plan - Ensemble Enhancement

## Phase 1: Kaggle ConvNeXt Training
- [x] Task: Set up Kaggle Notebook environment and dependencies.
    - [x] Subtask: Create a new Kaggle Notebook and install `timm`, `albumentations`, `facenet-pytorch`.
    - [x] Subtask: Configure `configs/kaggle_convnext.yaml` for `convnext_base` with aggressive blur/compression augmentations.
- [x] Task: Implement training script with "Blur-Breaker" augmentations.
    - [x] Subtask: Update `src/trainer/train.py` (or create `run_kaggle.py`) to include `Gaussian Blur`, `Motion Blur`, and `ImageCompression` in the training transforms.
    - [x] Subtask: Verify data loading from Kaggle input (`/kaggle/input/140k-real-and-fake-faces`).
- [ ] Task: Execute training on Kaggle.
    - [ ] Subtask: Run the training job for 10 epochs.
    - [ ] Subtask: Download the best checkpoint (`convnext_blur_best.pt`) to the local `checkpoints/` directory.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Kaggle ConvNeXt Training' (Protocol in workflow.md)

## Phase 2: TTA & Inference Logic
- [x] Task: Implement Test-Time Augmentation (TTA) in `src/trainer/inference.py`.
    - [x] Subtask: Add a `--tta` argument to the inference script.
    - [x] Subtask: Define TTA transforms (Original, Horizontal Flip, Sharpen, Blur, Compression).
    - [x] Subtask: Implement logic to average predictions across all augmented versions of an image.
- [ ] Task: Verify TTA logic.
    - [ ] Subtask: Run inference on a small subset of test data with and without TTA to confirm score stability.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: TTA & Inference Logic' (Protocol in workflow.md)

## Phase 3: Ensemble Construction
- [ ] Task: Generate inference results for ConvNeXt.
    - [ ] Subtask: Run inference on the full test set using the downloaded `convnext_blur_best.pt` with TTA enabled.
    - [ ] Subtask: Save results to `data/submission/submission_convnext_tta.csv`.
- [ ] Task: Create Ensemble V15.
    - [ ] Subtask: Update `scripts/ensemble.py` or `run_ensemble.bat` to combine DINO (10x Pseudo), ConvNeXt (Kaggle), and EfficientNet-B5.
    - [ ] Subtask: Experiment with weights (e.g., 0.5 DINO, 0.3 ConvNeXt, 0.2 EffB5) to generate `final_ensemble_v15.csv`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Ensemble Construction' (Protocol in workflow.md)