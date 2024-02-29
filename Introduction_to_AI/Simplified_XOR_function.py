import numpy as np
import matplotlib.pyplot as plt


# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Mean Squared Error loss function
def mse_loss(y_true, y_predicted):
    return 0.5 * ((y_true - y_predicted) ** 2)


# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Corresponding labels
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases randomly
np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights and biases for the hidden layer
W_1 = np.random.rand(input_neurons, hidden_neurons)
b_1 = np.random.rand(1, hidden_neurons)

# Weights and biases for the output layer
W_2 = np.random.rand(hidden_neurons, output_neurons)
b_2 = np.random.rand(1, output_neurons)

# Hyperparameters
learning_rate = 20
epochs = 8000

# Lists to store weights and biases
hidden_weights_history = []
hidden_biases_history = []
output_weights_history = []
output_biases_history = []

global y_pred

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    # Hidden layer
    Z_1 = np.dot(X, W_1) + b_1
    a_1 = sigmoid(Z_1)
    # Output layer
    Z_2 = np.dot(a_1, W_2) + b_2
    a_2 = sigmoid(Z_2)
    y_pred = a_2

    # Calculate loss
    loss = mse_loss(y, y_pred)

    # Backpropagation
    # Calculate the error at the output layer
    output_delta = (y_pred - y) * sigmoid_derivative(y_pred)
    hidden_error = output_delta.dot(W_2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a_1)

    # Update weights and biases
    W_2 -= a_1.T.dot(output_delta) * learning_rate
    b_2 -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W_1 -= X.T.dot(hidden_delta) * learning_rate
    b_1 -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Append weights and biases to history
    hidden_weights_history.append(W_1.copy())
    hidden_biases_history.append(b_1.copy())
    output_weights_history.append(W_2.copy())
    output_biases_history.append(b_2.copy())

# Convert history lists to numpy arrays
hidden_weights_history = np.array(hidden_weights_history)
hidden_biases_history = np.array(hidden_biases_history)
output_weights_history = np.array(output_weights_history)
output_biases_history = np.array(output_biases_history)

print(y_pred)
# Plot weights and biases over epochs
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
for i in range(W_1.shape[1]):
    plt.plot(hidden_weights_history[:, :, i], label=f'Hidden Neuron {i + 1}')
plt.title('Hidden Layer Weights')
plt.xlabel('Epochs')
plt.ylabel('Weights')
plt.legend()

plt.subplot(2, 2, 2)
for i in range(b_1.shape[0]):
    plt.plot(hidden_biases_history[:, i], label=f'Hidden Neuron {i + 1}')
plt.title('Hidden Layer Biases')
plt.xlabel('Epochs')
plt.ylabel('Biases')
plt.legend()

plt.subplot(2, 2, 3)
for i in range(W_2.shape[1]):
    plt.plot(output_weights_history[:, :, i], label=f'Output Neuron {i + 1}')
plt.title('Output Layer Weights')
plt.xlabel('Epochs')
plt.ylabel('Weights')
plt.legend()

plt.subplot(2, 2, 4)
for i in range(b_2.shape[1]):
    plt.plot(output_biases_history[:, i], label=f'Output Neuron {i + 1}')
plt.title('Output Layer Biases')
plt.xlabel('Epochs')
plt.ylabel('Biases')
plt.legend()

plt.tight_layout()
plt.show()
