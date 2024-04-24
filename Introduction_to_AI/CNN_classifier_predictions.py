import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

pretrained_model = tf.keras.models.load_model('CNN_classifier.h5', compile=False)
#########################
# Show model summary
pretrained_model.summary()
#########################


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_test = tf.keras.utils.to_categorical(y_test)

# Predictions
##########################
for i in range(x_test.shape[0]):
    print(x_test[i].shape)
    prediction = pretrained_model.predict(np.expand_dims(x_test[i], axis=0))
    prediction = np.argmax(prediction)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(x_test[i])
    plt.title(str(prediction))
    plt.show()
