import tensorflow as tf

model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

if __name__ == '__main__':
    model.save('keras/saved_models/vgg16.model')
