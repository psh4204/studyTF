import tensorflow as tf
import pandas as pd
import numpy as np
data = pd.read_csv('gpascore.csv')
data = data.dropna()
y데이터 = data['admit'].values
x데이터 = [ ]

for i, rows in data.iterrows():
    # iterrow() : 판다스 데이터프레임에서 한행씩 볼수있음.
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

## 대학원 합격, 불합격 여부
### admit 0 불합격
### gre 영어성적
### gpa 학점
### rank 1높음

# 딥러닝 모델 만들기 ( 히든레이어 )
## 딥러닝 노드 갯수는 마음대로 넣어도 됨 -> 실험적을 파악해야함 (관례적으로 2의 제곱수로 넣음)
### 결과는 하나기 때문에 하나만 적어두겠음
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'), #  tanH 그냥 넣음
    tf.keras.layers.Dense(1, activation='sigmoid') # sigmoid = 뱉는 값이 0.0~1.0 뿐
])

## 모델 완성단계
### optimizer 는 adam 쓰는게 일반적
### loss 는 0,1 값을 가져올거면 binary_corssentropy 
model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습시기키
## x에서는 학습데이터, y데이터는 실제정답
## 에포크는 데이터 보는 횟수
## numpy로 넣어야함
model.fit(np.array(x데이터), np.array(y데이터), epochs=1000)


# 예측
## 예측값 넣고 예측해보기 ( 누군가의 학점. 내학점 )
예측값 = model.predict([[750, 3.70,3],[750, 3.39, 3]])
print(예측값) # 지방대도 운좋게붙네.. 이런 ㅋㅋ
