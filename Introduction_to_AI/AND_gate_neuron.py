import random
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self, weights, bias, learning_rate=0.1):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.history_weights = [list(weights)]
        self.bias_history = [bias]

    def activate(self, inputs):
        assert len(inputs) == len(self.weights)

        weighted_sum = sum([inputs[i] * self.weights[i] for i in range(len(inputs))]) + self.bias

        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def update_weights(self, inputs, target):
        prediction = self.activate(inputs)
        error = target - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        self.bias += self.learning_rate * error
        self.bias_history.append(self.bias)
        self.history_weights.append(list(self.weights))


and_weights = [0, 0]
and_bias = 1  

and_gate = Neuron(and_weights, and_bias)

training_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

epochs = 300
weights_history = []
bias_history = []
for epoch in range(epochs):
    for input_data, target_output in training_data:
        and_gate.update_weights(input_data, target_output)
    weights_history.append(list(and_gate.history_weights))
    bias_history.append(and_gate.bias_history)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].set_title('AND Gate Neuron')
axs[0].set_xlabel('Input 1')
axs[0].set_ylabel('Input 2')

for input_data, target_output in training_data:
    if target_output == 1:
        axs[0].scatter(input_data[0], input_data[1], color='blue', label='Positive (1)')
    else:
        axs[0].scatter(input_data[0], input_data[1], color='red', label='Negative (0)')

axs[0].legend()
axs[0].grid(True)
temp = and_gate.history_weights[::4]
temp_bias = bias_history[0][::4]
# temp.pop(0)
for i, j in zip(range(epochs), range(len(and_weights))):
    axs[1].plot(range(epochs + 1), [w[j] for w in temp], label=f'Weight {j + 1}')
axs[1].plot(range(epochs + 1), [b for b in temp_bias], label='Bias')
axs[1].set_title("Weights and Bias")
axs[1].set_xlabel("Epoch")
axs[1].legend()
axs[1].grid(True)
plt.savefig('/home/abdalraheem/Desktop/AND_neuron.png')
plt.show()

print(and_gate.activate([0, 0]))  # Output: 0
print(and_gate.activate([0, 1]))  # Output: 0
print(and_gate.activate([1, 0]))  # Output: 0
print(and_gate.activate([1, 1]))  # Output: 1

