import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.accuracy_list = []
        self.loss_list = []
        self.iterations = 0 

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def calculate_accuracy(self, predicted, actual):
        correct = 0
        total = len(actual)
        
        for i in range(total):
            predicted_label = np.argmax(predicted[i])
            true_label = np.argmax(actual[i])
            if predicted_label == true_label:
                correct += 1
        
        accuracy = correct / total * 100
        
        return accuracy

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                
            err /= samples
            predicted = self.predict(x_train)
            accuracy = self.calculate_accuracy(predicted, y_train)

            self.loss_list.append(err)
            self.accuracy_list.append(accuracy)
            self.iterations += 1
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def graph_accuracy(self):
        plot_iterations = range(0, self.iterations)
        plt.plot(plot_iterations, self.accuracy_list)
        plt.title('Accuracy during training iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    def graph_loss(self):
        plot_iterations = range(0, self.iterations)
        plt.plot(plot_iterations, self.loss_list)
        plt.title('Loss during training iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
