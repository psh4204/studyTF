
import gc
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모형에 대한 
rootPath = os.getcwd()

imageGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1
)

from tensorflow.python.keras.models import load_model

model = load_model(rootPath+'ResNET50_Shape.h5')

model.summary()

testGenerator = ImageDataGenerator(
    rescale=1. / 255
)

testGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'Dataset', 'Shape_Dataset', 'Test'),
    target_size=(64, 64)
)

#model.evaluate(testGen)

from tensorflow.keras.preprocessing.image import array_to_img
import socket
import numpy as np

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8100))
server.listen(1)
print("Setting OK")
cls_index = ['circle  ', 'square  ', 'star    ', 'triangle']
while True:
    conn, addr = server.accept()
    cmnd = conn.recv(4)
    if 'PLAY' in str(cmnd):

        imgs = testGen.next()
        arr = imgs[0][0]
        #img = array_to_img(arr).resize((128, 128))
        #plt.imshow(img)
        #result = model.predict_classes(arr.reshape(1, 64, 64, 3))
        result = np.argmax(model.predict(arr.reshape(1, 64, 64, 3)), axis=-1)
        data = '{}'.format(cls_index[result[0]])
        conn.sendall(data.encode())
        gc.collect()
    elif 'QUIT' in str(cmnd):

        conn.sendall(b'QUIT')
        gc.collect()
        break

server.close()