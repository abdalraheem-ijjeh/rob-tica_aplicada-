import os
import numpy as np
import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

pretrained_model = tf.keras.models.load_model('ANN_model_MNIST_CLASSIFIER.h5')
#########################
# Show model summary
pretrained_model.summary()
#########################
