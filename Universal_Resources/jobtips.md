# Universal Deep Learning Development Guide

이 문서는 딥러닝 프로젝트 수행 시 성능 최적화, 코드 품질, 실험 효율성을 높이기 위한 보편적인 가이드라인입니다.

## 1. 실행 및 환경 (Execution & Environment)

### `accelerate launch` 활용 (Hugging Face Accelerate)
단일 GPU, 다중 GPU, TPU 환경을 코드 수정 없이 전환할 수 있습니다.
- **설정:** `accelerate config`
- **실행:** `accelerate launch train.py`
- **코드:**
  ```python
  from accelerate import Accelerator
  accelerator = Accelerator()
  model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
  accelerator.backward(loss) # loss.backward() 대체
  ```

### 시드 고정 (Reproducibility)
실험 결과의 재현성을 위해 난수 생성기의 시드를 고정해야 합니다.
```python
import torch
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 재현성 중요시 False, 속도 중요시 True

seed_everything(42)
```

## 2. 성능 최적화 (Performance Optimization)

### DataLoader 속도 향상
데이터 로딩이 학습 병목이 되지 않도록 설정합니다.
- `num_workers`: CPU 코어 수의 절반 또는 4~8 정도가 적당함. (Windows에서는 0이나 2 권장)
- `pin_memory=True`: CPU에서 GPU로 텐서 전송 속도 향상.
```python
DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

### CuDNN Benchmark
입력 이미지 크기가 고정되어 있다면 활성화하여 속도를 높입니다.
```python
torch.backends.cudnn.benchmark = True
```

### Mixed Precision (AMP) & Context Manager
메모리 절약 및 연산 가속을 위해 FP16을 사용합니다.
```python
# Accelerate 사용 시
with accelerator.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

## 3. 코드 품질 및 안전성 (Code Quality & Safety)

### Context Manager (`with`) 생활화
파일 입출력 등 리소스 관리는 반드시 `with` 구문을 사용합니다.
```python
# Good
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### Type Hinting
함수 정의 시 입출력 타입을 명시하여 가독성을 높이고 버그를 줄입니다.
```python
def train_one_epoch(model: torch.nn.Module, loader: DataLoader) -> float:
    ...
```

## 4. 디버깅 전략 (Debugging Strategy)

### 단일 배치 오버피팅 (Overfit on a Single Batch)
모델 구조나 로직에 문제가 없는지 확인하기 위해, **단 하나의 배치만 반복 학습**시켜 Loss가 0에 수렴하는지 확인합니다. 수렴하지 않으면 모델이나 데이터 파이프라인에 버그가 있는 것입니다.

### `detect_anomaly`
NaN 또는 Inf 발생 시 원인을 추적합니다. (디버깅 시에만 켜고, 실제 학습 시에는 끔 - 속도 저하)
```python
torch.autograd.set_detect_anomaly(True)
```