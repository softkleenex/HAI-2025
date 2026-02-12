# 🕵️‍♂️ HAI 2025 Deepfake Detection Challenge

![Rank](https://img.shields.io/badge/Rank-Top%209%25-brightgreen?style=flat-square&logo=kaggle)
![Private Score](https://img.shields.io/badge/Private%20AUC-0.7040-blue?style=flat-square)
![Public Score](https://img.shields.io/badge/Public%20AUC-0.7215-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

> **Competition:** [HAI 2025 Deepfake Detection Challenge](https://dacon.io/competitions/official/236628/overview/description) (Dacon)  
> **Team:** softkleenex  
> **Result:** 121st / 1,356 Teams (**Top 9%**)

## 📌 Overview
본 프로젝트는 생성형 AI로 만들어진 딥페이크 영상과 실제 영상을 구분하는 분류 모델을 개발하는 챌린지입니다.
다양한 생성 기법(FaceSwap, Diffusion 등)에 대응하기 위해 **Transformer(DINOv2)와 CNN(EfficientNet, ConvNeXt)의 하이브리드 앙상블**을 구축하였으며, 특히 Test Dataset의 분포 차이(Domain Gap)를 극복하기 위한 **Pseudo-Labeling (Semi-supervised Learning)** 전략에 집중했습니다.

---

## 🏆 Final Score
| Metric | Score | Rank | Note |
|:---:|:---:|:---:|:---|
| **Private AUC** | **0.70403** | **121위** (Top 9%) | 최종 순위 |
| **Public AUC** | **0.72151** | - | Pseudo-Labeling 적용 시 최고점 |

---

## 🛠️ Solution Strategy

### 1. Model Architecture (Ensemble)
서로 다른 귀납적 편향(Inductive Bias)을 가진 모델을 결합하여 예측의 안정성을 확보했습니다.

| Model | Architecture | Pretrained | Role | Weight |
|:---|:---|:---|:---|:---:|
| **DINOv2 Large** | ViT | ImageNet-21k | **[Main]** 전역적(Global) 특징 및 고수준 패턴 분석 | **0.5** |
| **EfficientNet-B5** | CNN | ImageNet-1k | **[Adapter]** 국소적(Local) 텍스처 및 아티팩트 탐지 | **0.4** |
| **ViT Base** | ViT | ImageNet-1k | **[Support]** 앙상블 다양성(Diversity) 강화 | **0.1** |
| **ConvNeXt** | CNN | ImageNet-1k | **[Generalist]** Blur/Compression 등 화질 저하에 강건함 | - |

### 2. Key Techniques
#### 🔹 Pseudo-Labeling (Test Set Injection)
*   **Problem:** 학습 데이터(FaceForensics++)와 테스트 데이터(Dacon Custom) 간의 도메인 차이 발생.
*   **Solution:** 
    1. 초기 앙상블 모델로 Test Set 추론.
    2. 신뢰도(Confidence)가 높은 상위/하위 5% 데이터 추출.
    3. 추출된 데이터를 학습 데이터(Train Set)에 **10배 Oversampling**하여 모델 재학습.
*   **Result:** Public Score **0.67 → 0.72**로 비약적 상승.

#### 🔹 Test-Time Augmentation (TTA)
*   추론 시 `Horizontal Flip`, `Gaussian Blur`, `Sharpen` 등 5가지 증강을 적용하고 결과를 평균(Average)하여 예측 안정성 확보.

#### 🔹 Blur-Breaker Training
*   저화질 Fake 영상이 Real로 오분류되는 현상을 막기 위해, 학습 시 `Albumentations`의 `ImageCompression`, `MotionBlur`를 강하게 적용하여 **"화질이 나빠도 가짜일 수 있음"**을 학습시킴.

---

## 📂 Project Structure
```text
/
├── checkpoints/       # 학습된 모델 가중치 (gitignore)
├── configs/           # 모델 및 실험 설정파일 (.yaml)
├── data/
│   ├── raw/           # 원본 데이터셋
│   ├── processed/     # 전처리된 얼굴 크롭 이미지
│   └── submission/    # 추론 결과 CSV
├── scripts/           # 자동화 및 유틸리티 스크립트
│   ├── run_kaggle.py  # Kaggle 학습 자동화
│   ├── ensemble.py    # 앙상블 스크립트
│   └── fix_submission.py # 제출 파일 형식 검증
├── src/
│   ├── data/          # Dataset & DataLoader (Custom Logic)
│   ├── models/        # DINO, EffNet, ConvNeXt 모델 정의
│   └── trainer/       # 학습 및 추론 루프
└── README.md          # 프로젝트 문서
```

---

## 📝 Experiments & Retrospective

| Ver | Method | Score (Public) | Insight |
|:---:|:---|:---:|:---|
| **V2** | Baseline Ensemble (DINO+ViT+EffB4) | 0.6781 | 기본적인 앙상블만으로도 단일 모델 대비 우수함. |
| **V7** | **Pseudo Labeling (3% Injection)** | **0.7038** | Test Set 도메인 적응이 점수 향상의 핵심임을 확인. |
| **V10** | **Pseudo (10x) + FewShot** | **0.7215** | 과적합과 일반화 사이의 최적 균형점 도달. (Best) |
| **V15** | ConvNeXt (Blur Augmentation) | 0.7193 | 화질 증강이 일부 도움되나, 라벨 노이즈 문제 발생 가능성. |
| **V17** | Video Model (CNN+LSTM) | 0.7124 | 프레임 단위 데이터셋 특성상 시계열 모델(LSTM)의 효과가 미미함. |

### 💡 What Worked
*   **Data-Centric Approach:** 모델 구조를 바꾸는 것보다, **데이터(Pseudo-Labeling)**를 어떻게 구성하느냐가 성능에 훨씬 큰 영향을 미침.
*   **Self-Supervised Learning:** 적은 데이터 환경에서는 DINOv2 같은 SSL 모델이 CNN보다 강력한 Feature Extractor 역할을 수행함.

### ⚠️ Challenges & Future Work
*   **Video Context:** 프레임 단위 분류에 집중하느라 영상 전체의 시계열적(Temporal) 부자연스러움을 완벽히 활용하지 못함. (`VideoMAE` 등 3D-CNN 도입 필요)
*   **External Data:** Celeb-DF 등 외부 데이터를 추가했으나, 도메인 갭으로 인해 성능 향상에는 기여하지 못함.

---

### 💻 Installation & Usage

**1. Environment Setup**
```bash
git clone https://github.com/softkleenex/HAI-2025.git
cd HAI-2025
pip install -r requirements.txt
```

**2. Inference**
```bash
# Run inference using the best checkpoint
python src/trainer/inference.py --config configs/dino_large.yaml --checkpoint checkpoints/dino_best.pt --output submission.csv --tta
```

**3. Ensemble**
```bash
# Combine predictions
python scripts/ensemble.py --inputs sub1.csv sub2.csv --weights 0.5 0.5 --output final.csv
```