import tensorflow as tf
import matplotlib.pyplot as plt

# trainX : 이미지데이터
# trainY : 정답 (라벨)
((trainX, trainY),(testX,testY)) = tf.keras.datasets.fashion_mnist.load_data()

# 라벨 이름들
class_name = ['T/shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']

# 모델 만들기
## 확률예측문제라면 
### 1. 일단 마지막 레이어 노드수를 카테고리 갯수만큼
### 2. cross entropy라는 loss 함수 사용
#### relu : 음수는 다 0으로, convolution layer에서 자주씀
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax") # sofmax :0~1 로 압축시켜줌_카테고리예측용_총합1 // sigmoid 는 정답,오답 두개분류일때 사용)
])

# 모델 아웃라인 출력하기
## 학습을 시키기전에 인풋의 요약을 보고싶으면 모델에 모양을 지정시켜야함(input_shape=(n,n))
### tf.keras.layers.Dense(128, input_shape=(28,28),activation="relu"),
model.summary()

## 다수의 카테고리중 어디 한곳에 들어갈지 정할때 (원핫인코딩) -> sparse_카테고리_크로스엔트로피
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10)

