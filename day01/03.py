import tensorflow as tf

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
modle.fit(x데이터, y데이터, epochs=1)
