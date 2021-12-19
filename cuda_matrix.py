from math import exp
from numba import cuda

"""Cuda kernels for neural network"""


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
    return 1 / (1 + exp(-x))


@cuda.jit(device=True)
def dsigmoid(y):
    return y * (1 - y)


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
def add(arr1, arr2):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        cuda.atomic.add(arr1, (x, y), arr2[x, y])
