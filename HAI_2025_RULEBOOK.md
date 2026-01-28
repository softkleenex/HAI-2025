# HAI 2025 Hecto AI Challenge: Deepfake Detection - Rulebook & Knowledge Base
https://dacon.io/competitions/official/236628/mysubmission
## 1. Competition Overview
*   **Goal:** Deepfake Detection (Binary Classification: Real vs. Fake).
*   **Metric:** **ROC-AUC** (with Sample Weights - weights are hidden).
*   **Schedule:**
    *   Start: 2025.12.29
    *   End: 2026.02.02 09:59
    *   Code Submission: 2026.02.09
*   **Host:** Hecto / **Platform:** DACON

## 2. Data Rules (Critical)
*   **Official Training Data:** **None provided.**
    *   Participants **must self-source** external data (e.g., FaceForensics++, FFHQ, Celeb-DF).
    *   External data must be legally usable (Copyright/Privacy compliant).
    *   Generated data (via API) is allowed if saved as static files.
*   **Test Data:**
    *   Provided: 500 images/videos (in `data/test`).
    *   **Usage:** **Inference ONLY.**
    *   **Strict Prohibition:** Using Test set for training, fine-tuning, or pseudo-labeling is **Data Leakage** (Disqualification).
*   **Input Format:**
    *   Video inputs must be decomposed into **Images (Frames)**.
    *   Model input must be **Image-level**.

## 3. Model Architecture Rules (Single Model Policy)
The competition enforces a **Single Model** policy with specific definitions.

### ✅ Allowed
*   **Sequential Pipeline:** `[Face Detector] -> [Classifier]` is allowed.
    *   The detector (e.g., RetinaFace) is considered preprocessing.
*   **Single Backbone with Multiple Heads:** One encoder shared by multiple heads (e.g., classification head + auxiliary head) is allowed.
*   **Pre-trained Models:**
    *   Must be publicly available before **2025.12.29**.
    *   Non-Commercial (NC) license architectures are allowed.

### 🚫 Disallowed (Strictly Prohibited)
*   **Ensembles:** Combining outputs (voting, averaging, weighted sum) from independent models.
*   **Parallel Backbones:** Combining features from multiple independent encoders (e.g., ResNet + EfficientNet in parallel).
*   **Temporal Learning:**
    *   Models **cannot** learn temporal relationships between frames.
    *   **No RNN, LSTM, GRU, or 3D-CNNs** that mix frame information inside the model.
*   **Test-Time Augmentation (TTA):** The rules explicitly list TTA as "Disallowed" in the context of "combining results from repeated inference" under the Single Model section (Needs careful handling, usually internal multi-crop is fine, but multi-inference averaging might be flagged).

## 4. Inference & Submission Rules
*   **Environment:**
    *   **Offline Only:** No internet access during inference.
    *   **Hardware:** L40S GPU (48GB VRAM), 16 vCPUs, 96GB RAM.
    *   **Time Limit:** Max **1 Hour** for the entire test set (preprocessing + inference).
*   **Post-Processing:**
    *   Frame-level predictions can be aggregated (e.g., averaged) **outside** the model to produce a video-level label.
*   **Submission Folder Structure:**
    ```text
    your_submission/
    ├── model/
    │   └── model.pt          # Single weight file
    ├── src/
    │   ├── models.py
    │   ├── dataset.py
    │   └── utils.py
    ├── config/
    │   └── config.yaml
    ├── env/
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── train_data/           # Sample of training data used
    ├── test_data/            # The provided test set
    ├── train.py
    ├── inference.py          # Entry point
    └── README.md
    ```

## 5. Q&A Insights (from Talks)
*   **Q:** Can I use a detector (preprocessing) and a classifier?
    *   **A:** Yes, sequential use is allowed.
*   **Q:** Can I share weights between heads?
    *   **A:** Yes, single backbone with multi-head is allowed.
*   **Q:** Is time-series info allowed?
    *   **A:** No direct learning of temporal info inside the model. Frame aggregation must happen post-inference.
