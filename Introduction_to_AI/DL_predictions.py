import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

pretrained_model = tf.keras.models.load_model('ANN_model_MNIST_CLASSIFIER.h5', compile=False)
#########################
# Show model summary
pretrained_model.summary()
#########################
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape, "test samples")

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

##########################

##########################
for i in range(x_test.shape[0]):
    print(x_test[i].shape)
    prediction = pretrained_model.predict(np.expand_dims(x_test[i], axis=0))
    prediction = np.argmax(prediction)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(x_test[i])
    plt.title(str(prediction))
    plt.show()
