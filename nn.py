from math import exp, tanh, ceil
from random import shuffle
import warnings

import numpy as np
from mnist import MNIST
from numba import cuda, core

import matrix
import cuda_matrix as cm


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
        self.tpb = (8, 8)
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
        self.train_data = np.matrix([[i / 255 for i in image] for image in self.train_data]).T
        print('.', end='')
        self.labels = np.matrix(self.train_labels)
        self.train_labels = np.zeros((self.weights[-1].shape[0], self.train_data.shape[1]))
        for i in range(self.train_labels.shape[1]):
            self.train_labels[self.labels[0, i], i] = 1
        print("Done")
        self.dataset_loaded = True

    def feedforward(self, inputs):
        hidden = np.add(
            np.matmul(self.weights[0], inputs),
            self.biases[0])
        matrix.map(self.activation[0], hidden)

        output = np.add(
            np.matmul(self.weights[1], hidden),
            self.biases[1])
        matrix.map(self.activation[0], output)
        return hidden, output

    def feedforward_np(self, inputs):
        return cm.feedforward_np(
            cm.feedforward_np(inputs, self.weights[0], self.biases[0]),
            self.weights[1],
            self.biases[1]
        )

    def feedforward_cuda(self, inputs):

        inputs_d = cuda.to_device(inputs)
        weights_d = cuda.to_device(self.weights[0])
        biases_d = cuda.to_device(self.biases[0])
        outputs_d = cuda.device_array((self.weights[0].shape[0], inputs.shape[1]))

        cm.feedforward_step[get_blocks_per_grid(outputs_d.shape, self.tpb), self.tpb](inputs_d, weights_d, biases_d, outputs_d)
        cuda.synchronize()

        inputs_d = outputs_d
        weights_d = cuda.to_device(self.weights[1])
        biases_d = cuda.to_device(self.biases[1])
        outputs_d = cuda.device_array((self.weights[1].shape[0], inputs.shape[1]))

        cm.feedforward_step[get_blocks_per_grid(outputs_d.shape, self.tpb), self.tpb](inputs_d, weights_d, biases_d, outputs_d)
        cuda.synchronize()

        return outputs_d.copy_to_host()

    def max_class(self, inputs):
        output = list(self.feedforward_cuda(inputs))
        return output.index(max(output))

    def binary_class(self, inputs):
        return int(self.feedforward(inputs)[1][0, 0] >= 0.5)

    def test_accuracy(self):
        output = self.feedforward_cuda(self.train_data)
        print(f"Accuracy: {round((output.argmax(axis=0) == self.labels[0, :]).sum() / self.train_data.shape[1] * 100, 5)}%")

    def backpropogation(self, inputs, target, lr):
        hidden, output = self.feedforward(inputs)

        error = np.subtract(target, output)
        hidden_error = np.matmul(self.weights[1].T, error)

        matrix.map(self.activation[1], output)
        output = (np.multiply(output, error) * lr)

        for i in range(output.shape[1]):
            self.weights[1] += np.matmul(output[:, i], hidden.T[i, :])
        self.biases[1] += output.sum(axis=1)

        matrix.map(self.activation[1], hidden)
        hidden = (np.multiply(hidden, hidden_error) * lr)

        for i in range(hidden.shape[1]):
            self.weights[0] += np.matmul(hidden[:, i], inputs.T[i, :])
        self.biases[0] += hidden.sum(axis=1)

    def backpropogation_cuda(self, inputs, targets, lr):
        inputs_d = cuda.to_device(inputs)
        weights_d = cuda.to_device(self.weights[0])
        biases_d = cuda.to_device(self.biases[0])
        hidden_d = cuda.device_array((self.weights[0].shape[0], inputs.shape[1]))

        cm.feedforward_step[get_blocks_per_grid(hidden_d.shape, self.tpb), self.tpb](inputs_d, weights_d, biases_d, hidden_d)
        cuda.synchronize()

        inputs_d = hidden_d
        weights_d = cuda.to_device(self.weights[1])
        biases_d = cuda.to_device(self.biases[1])
        outputs_d = cuda.device_array((self.weights[1].shape[0], inputs.shape[1]))

        cm.feedforward_step[get_blocks_per_grid(outputs_d.shape, self.tpb), self.tpb](inputs_d, weights_d, biases_d, outputs_d)
        cuda.synchronize()

        errors_d = cuda.device_array(targets.shape)
        targets_d = cuda.to_device(targets)
        hidden_errors_d = cuda.device_array((self.weights[1].shape[1], errors_d.shape[1]))

        cm.subtract[get_blocks_per_grid(errors_d.shape, self.tpb), self.tpb](targets_d, outputs_d, errors_d)
        cuda.synchronize()

        cm.matmul[get_blocks_per_grid(hidden_errors_d.shape, self.tpb), self.tpb](weights_d.transpose(), errors_d, hidden_errors_d)
        cuda.synchronize()

        gradient_d = cuda.device_array(outputs_d.shape)
        cm.gradient[get_blocks_per_grid(gradient_d.shape, self.tpb), self.tpb](outputs_d, errors_d, lr, gradient_d)
        cuda.synchronize()

        delta_b_d = cuda.device_array(self.biases[1].shape)
        cm.sum_cols[ceil(self.biases[1].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()

        self.biases[1] += delta_b_d.copy_to_host()

        delta_w_d = cuda.device_array((gradient_d.shape[0], hidden_d.shape[0]))
        cm.matmul[get_blocks_per_grid(self.weights[1].shape, self.tpb), self.tpb](gradient_d, hidden_d.transpose(), delta_w_d)
        cuda.synchronize()

        self.weights[1] += delta_w_d.copy_to_host()

        gradient_d = cuda.device_array(hidden_d.shape)
        cm.gradient[get_blocks_per_grid(gradient_d.shape, self.tpb), self.tpb](hidden_d, hidden_errors_d, lr, gradient_d)
        cuda.synchronize()

        delta_b_d = cuda.device_array(self.biases[0].shape)
        cm.sum_cols[ceil(self.biases[0].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()

        self.biases[0] += delta_b_d.copy_to_host()

        delta_w_d = cuda.device_array((gradient_d.shape[0], inputs.shape[0]))
        inputs_d = cuda.to_device(inputs.T)
        cm.matmul[get_blocks_per_grid(self.weights[0].shape, self.tpb), self.tpb](gradient_d, inputs_d, delta_w_d)
        cuda.synchronize()

        self.weights[0] += delta_w_d.copy_to_host()

    def train_batch(self, repeat, lr, batch_size):
        for r in range(repeat):
            print(f"\r{r + 1} / {repeat}", end='')
            perm = np.random.permutation(self.train_labels.shape[1])
            for i in range(ceil(perm.size / batch_size)):
                batch_perm = perm[i * batch_size : (i + 1) * batch_size]
                self.backpropogation_cuda(self.train_data[:, batch_perm], self.train_labels[:, batch_perm], lr)
        print()

    def export_weights(self):
        pass

    def load_weights(self):
        pass


if __name__ == '__main__':
    warnings.simplefilter('ignore', category=core.errors.NumbaPerformanceWarning)
    dc = NeuralNetwork(784, 64, 10)
    dc.load_dataset('dataset')
    # dc.train_data = np.matrix([[0, 0],
    #                            [1, 1],
    #                            [0, 1],
    #                            [1, 0]]).T
    # dc.train_labels = np.array([[0, 0, 1, 1]])
    # dc.dataset_loaded = True

    while True:
        dc.test_accuracy()
        dc.train_batch(1, 0.00001, 1024)
