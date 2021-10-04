import tensorflow as tf
import numpy as np

# trainX : 이미지데이터
# trainY : 정답 (라벨)
(trainX, trainY),(testX,testY) = tf.keras.datasets.fashion_mnist.load_data()

# (선택사항)이미지데이터 전처리 (0~1로 압축해서 넣는다.)
trainX = trainX/255.0
testX = testX/255.0

# 괄호처리(모양지정)을 해줘야함(넘파이 참 좋다)
## 28개, 28개, 60000개 있고, 컨볼루션하기위해 한차원 더만듦
trainX = trainX.reshape((trainX.shape[0], 28,28, 1))
testX = testX.reshape((testX.shape[0], 28,28, 1))


# 라벨 이름들
class_name = ['T/shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']

# 모델 만들기
## Flatten 을 사용하면 (1차원으로 해체) 응용력이 없어짐. ( 조금만 달라도 틀렸다고 나옴)
## 해결책 -> 컨볼루션 레이어
### feature Extraction : 특성 추출 : 이미지 인식에서는 필수.
### 1. 이미지에서 중요한 부분들을 추려서 복사본 20장 만든다
### 2. 거기엔 이미지의 중요한 feature, 특성이 담겨있음
### 3. 이걸로 학습한다.
model = tf.keras.Sequential([
    # 컨볼루션 레이어 생성
    ## 32개 컨볼루션 이미지와 커널사이즈는 3x3, 패딩(가에 공간)은 넣는게 좋음,
    ## relu는 음수를 안넣기 위해서,
    ## input_shpae() = ndim 에러를 없애기 위해 모양 지정 + 1차원 더 넣어줘야함
    ## 컬러데이터면 inputshape( , , 3) 이 되야겠죠 (R,G,B)
    tf.keras.layers.Conv2D(32, (3,3), padding="same",activation='relu', input_shape=(28,28,1)),
    ### 풀링하기. 맥스로 2x2 사이즈로(중앙으로 압축)
    tf.keras.layers.MaxPooling2D( (2,2) ),
    # tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax") # sofmax :0~1 로 압축시켜줌_카테고리예측용_총합1 // sigmoid 는 정답,오답 두개분류일때 사용)
])

# 모델 아웃라인 출력하기
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)


# 모델 평가
## 컴퓨터가 처음보는 데이터를 넣어줘야한다.
score = model.evaluate(testX, testY)
print(score)