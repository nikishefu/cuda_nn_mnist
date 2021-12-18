from numba import cuda
import numpy as np
from math import exp, ceil, tanh
from time import time


@cuda.jit(device=True)
def matmul_device(a, b):
    i, batch = cuda.grid(2)
    if i < a.shape[0] and batch < b.shape[1]:
        tmp = 0.
        for j in range(a.shape[1]):
            tmp += a[i, j] * b[j, batch]
        return tmp


@cuda.jit
def matmul(arr1, arr2, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(arr1.shape[1]):
            tmp += arr1[i, k] * arr2[k, j]
        out[i, j] = tmp


@cuda.jit(device=True)
def sigmoid(x):
    return tanh(x)
    # return 1 / (1 + exp(-x))


@cuda.jit(device=True)
def dsigmoid(y):
    return 1 - (y ** 2)
    # return y * (1 - y)


def sigmoid_host(arr):
    res = np.empty(arr.shape)
    x, y = arr.shape
    for i in range(x):
        for j in range(y):
            res[i, j] = 1 / (1 + exp(-arr[i, j]))
    return res


@cuda.jit
def feedforward_step(inputs, weights, biases, outputs):
    x, batch = cuda.grid(2)
    if x < outputs.shape[0] and batch < outputs.shape[1]:
        outputs[x, batch] = matmul_device(weights, inputs)
        cuda.atomic.add(outputs, (x, batch), biases[x, 0])
        outputs[x, batch] = sigmoid(outputs[x, batch])


@cuda.jit
def gradient(outputs, errors, lr, gradients):
    x, batch = cuda.grid(2)
    if x < outputs.shape[0] and batch < outputs.shape[1]:
        gradients[x, batch] = dsigmoid(outputs[x, batch]) * errors[x, batch] * lr

@cuda.jit
def sum_cols(in_arr, out):
    x = cuda.grid(1)
    if x < in_arr.shape[0]:
        out[x, 0] = 0
        for y in range(in_arr.shape[1]):
            cuda.atomic.add(out, (x, 0), in_arr[x, y])


@cuda.jit
def subtract(arr1, arr2, out):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        out[x, y] = arr1[x, y] - arr2[x, y]


@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        an_array[x, y] = (an_array[x, y] ** 100) ** 0.01


@cuda.jit
def multiply(arr1, arr2):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        arr1[x, y] *= arr2[x, y]


@cuda.jit
def add(arr1, arr2):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        cuda.atomic.add(arr1, (x, y), arr2[x, y])


def feedforward_np(inputs, weights, biases):
    return sigmoid_host(np.add(np.matmul(weights, inputs), biases))


if __name__ == '__main__':
    m1 = np.random.rand(1, 4)
    m2 = np.random.rand(1, 4)
    print(m1)
    print(m2)
    m1d = cuda.to_device(m1)
    m2d = cuda.to_device(m2)

    threadsperblock = (16, 16)
    blockspergrid_x = ceil(m1d.shape[0] / threadsperblock[0])
    blockspergrid_y = ceil(m2d.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    subtract[blockspergrid, threadsperblock](m2d, m1d)
    print(m2d.copy_to_host())
    print(m2 - m1)
