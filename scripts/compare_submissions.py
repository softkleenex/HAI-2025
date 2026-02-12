import pandas as pd
import numpy as np
import os

# 파일 경로 설정
v10_path = 'data/submission/final_ensemble_v10.csv'
v14_path = 'data/submission/final_ensemble_v14.csv'

# 데이터 로드
df10 = pd.read_csv(v10_path)
df14 = pd.read_csv(v14_path)

# 데이터 병합
merged = pd.merge(df10, df14, on='filename', suffixes=('_v10', '_v14'))

# 차이 계산
merged['diff'] = merged['prob_v10'] - merged['prob_v14']
merged['abs_diff'] = merged['diff'].abs()

# 1. 상관관계 확인
correlation = merged['prob_v10'].corr(merged['prob_v14'])
print(f"📉 두 모델 간 상관계수 (Correlation): {correlation:.4f}")

# 2. 판단이 뒤집힌 경우 (Threshold 0.5 기준)
# v10은 가짜(>0.5)라고 했는데, v14는 진짜(<0.5)라고 한 경우
v10_fake_v14_real = merged[(merged['prob_v10'] > 0.6) & (merged['prob_v14'] < 0.4)]

# v10은 진짜(<0.5)라고 했는데, v14는 가짜(>0.5)라고 한 경우
v10_real_v14_fake = merged[(merged['prob_v10'] < 0.4) & (merged['prob_v14'] > 0.6)]

print(f"\n🔄 [Decision Flip Analysis] (Threshold 0.5, with margin)")
print(f"1️⃣ V10(Fake) -> V14(Real): {len(v10_fake_v14_real)} 건")
if len(v10_fake_v14_real) > 0:
    print(v10_fake_v14_real[['filename', 'prob_v10', 'prob_v14']].head(5))

print(f"\n2️⃣ V10(Real) -> V14(Fake): {len(v10_real_v14_fake)} 건")
if len(v10_real_v14_fake) > 0:
    print(v10_real_v14_fake[['filename', 'prob_v10', 'prob_v14']].head(5))

# 3. 가장 큰 차이가 나는 Top 5
top_diff = merged.sort_values('abs_diff', ascending=False).head(5)
print(f"\n📊 [Top 5 Most Disagreed Images]")
print(top_diff[['filename', 'prob_v10', 'prob_v14', 'diff']])
