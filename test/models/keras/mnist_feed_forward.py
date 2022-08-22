from keras import Sequential
from keras.layers import Dense, Activation, Dropout

from keras.datasets import mnist
from keras.utils import np_utils

model = Sequential(
    [
        Dense(512, input_shape=(784,)),
        Activation('relu'),
        Dropout(0.2),

        Dense(512),
        Activation('relu'),
        Dropout(0.2),

        Dense(10),
        Activation('softmax')
    ]
)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=20,
        verbose=2,
        validation_split=0.1
    )

    model.save('keras/saved_models/my_ffn.model')

    print(model.summary())

    #summary_of_training(history)

    pass

