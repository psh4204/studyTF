
from logging import root
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
import tensorflow.keras

# 위노그라드 알고리즘 설정
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

CWD = os.getcwd()
rootPath = os.path.join(CWD, 'Dataset', 'Shape_Dataset')


# Train 폴더에서 Train 과 Validation 전부 함.
imageGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.25
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'Train'),
    target_size=(64, 64),
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'Train'),
    target_size=(64, 64),
    subset='validation'
)

model = Sequential()
model.add(ResNet50(include_top=True, weights=None, input_shape=(64, 64, 3), classes=3))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

# 체크포인트
saveCkpt = tf.keras.callbacks.ModelCheckpoint(
    # filepath='ckpt/mnnist{epoch}',
    filepath='ckpt/mnnist',
    monitor='val_acc',
    mode='max',
    save_weights_only=True,
    save_freq='epoch'
)

# 에포크는 데이터수가 너무적으면 많이 못넣음.
epochs = 30
for i in range(0,10000) :
    print("Shape 학습 횟수 : "+str(i))
    history = model.fit(
        trainGen,
        validation_data = validationGen,
        epochs=epochs,
        callbacks=[saveCkpt]
    )
    # 모델 세이브 (.h5)
    print(rootPath+'ResNET50_Shape.h5 저장')
    model.save(rootPath+'ResNET50_Shape.h5')