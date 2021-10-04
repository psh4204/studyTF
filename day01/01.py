from six import b
import tensorflow as tf

키 = 170
신발 = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

신발 = 키 * a + b

opt = tf.keras.optimizers.Adam(learning_rate=0.01) # 경사하강법을 도와주는 고마운 친구


def 손실함수() :
    # return (실제값 - 예측값)^2
    예측값 = 키 * a + b
    return tf.square(260 - 예측값)

# 경사하강법으로 w 값 구하기.
for i in range(300):
    opt.minimize(손실함수, var_list=[a,b]) # 경사하강법으로 업데이트 할 weight Variable 목록
    print(a.numpy() ,b.numpy())

