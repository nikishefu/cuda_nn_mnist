import warnings
from math import exp, tanh, ceil
from time import time

import numpy as np
from mnist import MNIST
from numba import cuda, core

import cuda_matrix as cm


class NeuralNetwork:
    sigmoid = (lambda x: 1 / (1 + exp(-x)), lambda y: y * (1 - y))
    tanh = (lambda x: tanh(x), lambda y: 1 - (y ** 2))

    def __init__(self, in_nodes, hid_nodes, out_nodes):

        # Each column of data is a single array of input
        # (e.g. a single image from MNIST)
        self.train_data = None
        self.test_data = None

        # Labels for corresponding data
        self.train_labels = None
        self.test_labels = None

        # Desired output for corresponding data based on self.train_labels
        self.train_targets = None

        # Threads per block for cuda
        self.tpb = (8, 8)

        self.weights = [
            cuda.to_device(np.random.rand(hid_nodes, in_nodes) * 2 - 1),
            cuda.to_device(np.random.rand(out_nodes, hid_nodes) * 2 - 1)]

        self.biases = [
            cuda.to_device(np.random.rand(hid_nodes, 1) * 2 - 1),
            cuda.to_device(np.random.rand(out_nodes, 1) * 2 - 1)]

        self.dataset_loaded = False

    def get_grid_dim(self, out_shape):
        """Returns cuda grid dimensions needed for kernel execution"""
        return (ceil(out_shape[0] / self.tpb[0]),
                ceil(out_shape[1] / self.tpb[1])), self.tpb

    def load_mnist(self, ds_path):
        """Loads MNIST dataset of training and testing data"""

        print("Loading MNIST dataset .", end='')
        self.train_data, self.train_labels = MNIST(ds_path).load_training()
        self.test_data, self.test_labels = MNIST(ds_path).load_testing()
        print('.', end='')
        self.train_data = np.matrix([[i / 255 for i in image]
                                     for image in self.train_data]).T
        self.test_data = np.matrix([[i / 255 for i in image]
                                    for image in self.test_data]).T
        print('.', end='')
        self.train_labels = np.matrix(self.train_labels)
        self.test_labels = np.matrix(self.test_labels)

        self.train_targets = np.zeros((self.weights[-1].shape[0],
                                       self.train_data.shape[1]))
        for i in range(self.train_targets.shape[1]):
            self.train_targets[self.train_labels[0, i], i] = 1
        self.dataset_loaded = True
        print(" Done")

    def feedforward_cuda(self, inputs):
        """Returns outputs of NN for each input"""

        inputs_d = cuda.to_device(inputs)
        outputs_d = cuda.device_array((self.weights[0].shape[0],
                                       inputs.shape[1]))

        cm.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[0], self.biases[0], outputs_d)
        cuda.synchronize()

        inputs_d = outputs_d
        outputs_d = cuda.device_array((self.weights[1].shape[0],
                                       inputs.shape[1]))

        cm.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[1], self.biases[1], outputs_d)
        cuda.synchronize()

        return outputs_d.copy_to_host()

    def test_accuracy(self):
        """Returns categorical accuracy of NN on self.test_data"""

        output = self.feedforward_cuda(self.test_data)
        return (output.argmax(axis=0) == self.test_labels[0, :]).sum() / self.test_data.shape[1] * 100

    def train_accuracy(self):
        """Returns categorical accuracy of NN on self.train_data"""

        output = self.feedforward_cuda(self.train_data)
        return (output.argmax(axis=0) == self.train_labels[0, :]).sum() / self.train_data.shape[1] * 100

    def print_accuracy(self):
        print(f"Test accuracy: {round(self.test_accuracy(), 2)}%")
        print(f"Train accuracy: {round(self.train_accuracy(), 2)}%")

    def backpropogation_cuda(self, inputs, targets, lr):
        """
        Implements batch gradient descent.
        Each column of targets is desired output of NN for each input
        """
        inputs_d = cuda.to_device(inputs)
        hidden_d = cuda.device_array((self.weights[0].shape[0], inputs.shape[1]))

        cm.feedforward_step[self.get_grid_dim(hidden_d.shape)](inputs_d, self.weights[0], self.biases[0], hidden_d)
        cuda.synchronize()

        inputs_d = hidden_d
        outputs_d = cuda.device_array((self.weights[1].shape[0], inputs.shape[1]))

        cm.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[1], self.biases[1], outputs_d)
        cuda.synchronize()

        errors_d = cuda.device_array(targets.shape)
        targets_d = cuda.to_device(targets)
        hidden_errors_d = cuda.device_array((self.weights[1].shape[1], errors_d.shape[1]))

        cm.subtract[self.get_grid_dim(errors_d.shape)](targets_d, outputs_d, errors_d)
        cuda.synchronize()

        cm.matmul[self.get_grid_dim(hidden_errors_d.shape)](self.weights[1].transpose(), errors_d, hidden_errors_d)
        cuda.synchronize()

        gradient_d = cuda.device_array(outputs_d.shape)
        cm.gradient[self.get_grid_dim(gradient_d.shape)](outputs_d, errors_d, lr, gradient_d)
        cuda.synchronize()

        delta_b_d = cuda.device_array(self.biases[1].shape)
        cm.sum_cols[ceil(self.biases[1].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()

        cm.add[self.get_grid_dim(self.biases[1].shape)](self.biases[1], delta_b_d)

        delta_w_d = cuda.device_array((gradient_d.shape[0], hidden_d.shape[0]))
        cm.matmul[self.get_grid_dim(self.weights[1].shape)](gradient_d, hidden_d.transpose(), delta_w_d)
        cuda.synchronize()

        cm.add[self.get_grid_dim(self.weights[1].shape)](self.weights[1], delta_w_d)

        gradient_d = cuda.device_array(hidden_d.shape)
        cm.gradient[self.get_grid_dim(gradient_d.shape)](hidden_d, hidden_errors_d, lr, gradient_d)
        cuda.synchronize()

        delta_b_d = cuda.device_array(self.biases[0].shape)
        cm.sum_cols[ceil(self.biases[0].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()

        cm.add[self.get_grid_dim(self.biases[0].shape)](self.biases[0], delta_b_d)

        delta_w_d = cuda.device_array((gradient_d.shape[0], inputs.shape[0]))
        inputs_d = cuda.to_device(inputs.T)
        cm.matmul[self.get_grid_dim(self.weights[0].shape)](gradient_d, inputs_d, delta_w_d)
        cuda.synchronize()

        cm.add[self.get_grid_dim(self.weights[0].shape)](self.weights[0], delta_w_d)
        cuda.synchronize()

    def train(self, epochs: int, learning_rate: float, batch_size: int):
        """
        Train neural network on a dataset, divided into batches.
        Weights are corrected after each batch
        """
        for r in range(epochs):
            start_time = time()

            perm = np.random.permutation(self.train_targets.shape[1])
            batch_count = ceil(perm.size / batch_size)
            for i in range(batch_count):
                pr_bar = ('#' * ((i + 1) * 20 // batch_count)).ljust(20, ' ')
                dur = round(time() - start_time, 1)
                print(f"\rEpoch: {r + 1} / {epochs} [{pr_bar}] {dur}s", end='')

                batch_perm = perm[i * batch_size: (i + 1) * batch_size]
                self.backpropogation_cuda(self.train_data[:, batch_perm],
                                          self.train_targets[:, batch_perm],
                                          learning_rate)

            print()
            self.print_accuracy()
            print()

    def save_weights(self, filename: str):
        """Save weights to weights directory as an .npz archive"""
        np.savez_compressed(f"weights/{filename}",
                            *[self.weights[i].copy_to_host()
                              for i in range(len(self.weights))],
                            *[self.biases[i].copy_to_host()
                              for i in range(len(self.biases))])

    def load_weights(self, filename: str):
        """Load weights stored as an .npz archive from weights directory"""
        loaded = np.load(f"weights/{filename}.npz")
        self.weights = [loaded['arr_0'], loaded['arr_1']]
        self.biases = [loaded['arr_2'], loaded['arr_3']]


if __name__ == '__main__':
    # Silence Numba warnings about low occupancy of GPU
    warnings.simplefilter('ignore',
                          category=core.errors.NumbaPerformanceWarning)

    dc = NeuralNetwork(784, 64, 10)
    dc.load_mnist('dataset')

    # # Dataset for xor problem
    # dc.train_data = np.matrix([[0, 0],
    #                            [1, 1],
    #                            [0, 1],
    #                            [1, 0]]).T
    # dc.train_labels = np.array([[0, 0, 1, 1]])
    # dc.dataset_loaded = True

    dc.print_accuracy()
    print()

    # dc.load_weights('weights1')
    # dc.print_accuracy()

    dc.train(5, 0.01, 1024)
    # dc.save_weights('weights1')
