from layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size, weight_decay=0.001):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        self.weight_decay = weight_decay

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        weights_regularization = 2 * self.weight_decay * self.weights

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        weights_error += weights_regularization

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        return input_error