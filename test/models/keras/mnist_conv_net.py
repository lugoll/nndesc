from keras import Sequential, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np

model = Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #X_train = X_train.reshape(60000, 784)
    #X_test = X_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=1,
        verbose=2,
        validation_split=0.1
    )

    model.save('keras/saved_models/my_cnn.model')

    #onnx_model = keras2onnx.convert_keras(model, 'my_cnn', target_opset=13, )
    #keras2onnx.save_model(onnx_model, 'my_cnn.onnx')

    print(model.summary())

    #summary_of_training(history)

    pass
