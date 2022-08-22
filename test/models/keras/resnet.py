import tensorflow as tf

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)


if __name__ == '__main__':
    model.save('keras/saved_models/resnet.model')