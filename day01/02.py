import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

# 모델만들기
a = tf.Variable(0.1)
b = tf.Variable(0.1)

# 학습
def 손실함수(a,b):
    # 텐서상에서는 리스트를 for문 으로 돌리면 우리가 생각했던 대로 리스트 속 값들 하나하나 차례로 만져짐
    예측_y = train_x * a + b
    # mean Suared Error = 평균((예측값-실제)^2 + n ...)
    return tf.keras.losses.mse(train_y, 예측_y) #실제값 , 예측값

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(3000):
    opt.minimize(lambda:손실함수(a,b), var_list=[a,b])
    print(a.numpy(), b.numpy())