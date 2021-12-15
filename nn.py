from random import shuffle
from math import exp, tanh, ceil
import numpy as np
from mnist import MNIST
import matrix


def get_blocks_per_grid(out_shape, tpb):
    return ceil(out_shape[0] / tpb[0]), ceil(out_shape[1] / tpb[1])


class NeuralNetwork:
    sigmoid = (lambda x: 1 / (1 + exp(-x)), lambda y: y * (1 - y))
    tanh = (lambda x: tanh(x), lambda y: 1 - (y ** 2))

    def __init__(self, in_nodes, hid_nodes, out_nodes):

        self.train_data = None
        self.train_labels = None

        self.learning_rate = 0.2
        self.activation = self.sigmoid
        if out_nodes == 1:
            self.get_label = self.binary_class
        else:
            self.get_label = self.max_class

        self.weights = [
            np.random.rand(hid_nodes, in_nodes) * 2 - 1,
            np.random.rand(out_nodes, hid_nodes) * 2 - 1]

        self.biases = [
            np.random.rand(hid_nodes, 1) * 2 - 1,
            np.random.rand(out_nodes, 1) * 2 - 1]

        self.dataset_loaded = False

    def load_dataset(self, ds_path):
        print("Loading dataset .", end='')
        self.train_data, self.train_labels = MNIST(ds_path).load_training()  # TODO test data
        print('.', end='')
        self.train_data = [np.matrix([[i / 255 for i in image]]).T for image in self.train_data]
        print('. Done')
        self.dataset_loaded = True

    def feedforward(self, inputs):
        hidden = matrix.add(
            matrix.matmul(self.weights[0], inputs),
            self.biases[0])
        matrix.map(self.activation[0], hidden)

        output = matrix.add(
            matrix.matmul(self.weights[1], hidden),
            self.biases[1])
        matrix.map(self.activation[0], output)
        return hidden, output

    def max_class(self, inputs):
        output = list(self.feedforward(inputs)[1])
        return output.index(max(output))

    def binary_class(self, inputs):
        return int(self.feedforward(inputs)[1][0, 0] >= 0.5)

    def test_accuracy(self):
        correct = 0
        test_count = len(self.train_data)
        for i in range(test_count):
            correct += int(self.get_label(self.train_data[i]) == self.train_labels[i])
        print(f"Accuracy: {correct / test_count * 100}%")

    def backpropogation(self, inputs, target, lr):
        hidden, output = self.feedforward(inputs)

        error = matrix.subtract(target, output)

        matrix.map(self.activation[1], output)
        output = matrix.multiply(output, error, lr)

        self.weights[1] = matrix.add(self.weights[1], matrix.matmul(output, hidden.T))
        self.biases[1] = matrix.add(self.biases[1], output)

        hidden_error = matrix.matmul(self.weights[1].T, error)

        matrix.map(self.activation[1], hidden)
        hidden = matrix.multiply(hidden, hidden_error, lr)

        self.weights[0] = matrix.add(self.weights[0], matrix.matmul(hidden, inputs.T))
        self.biases[0] = matrix.add(self.biases[0], hidden)

    def train(self, repeat, lr):
        for r in range(repeat):
            print(f"\r{r + 1} / {repeat}", end='')
            indexes = list(range(len(self.train_data)))
            shuffle(indexes)
            for i in indexes:
                target = np.zeros((self.weights[-1].shape[0], 1))
                if target.shape[0] == 1:
                    target[0, 0] = self.train_labels[i]
                else:
                    target[self.train_labels[i], 0] = 1
                self.backpropogation(self.train_data[i], target, lr)

    def export_weights(self):
        pass

    def load_weights(self):
        pass


if __name__ == '__main__':
    dc = NeuralNetwork(784, 64, 10)
    dc.load_dataset('dataset')
    # dc.train_data = [np.matrix([[0, 0]]).T,
    #                  np.matrix([[1, 1]]).T,
    #                  np.matrix([[0, 1]]).T,
    #                  np.matrix([[1, 0]]).T]
    # dc.train_labels = [0, 0, 1, 1]
    # dc.dataset_loaded = True

    dc.train(100, 0.08)

    dc.test_accuracy()
