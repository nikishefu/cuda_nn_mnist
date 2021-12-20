# cuda_nn_mnist

Neural network library for digit classification powered by cuda

## Resources
The library was built to work with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
[python-mnist](https://pypi.org/project/python-mnist/) data parser was used to load the dataset.

## Hardware requirements
The library uses [Numba](https://numba.readthedocs.io/en/stable/cuda/index.html) which
[supports](http://numba.pydata.org/numba-doc/latest/cuda/overview.html#supported-gpus)
CUDA-enabled GPUs with compute capability 2.0 or above with an up-to-date Nvidia driver.
See the list of [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus)

## Trained weights
Trained weights are located in `weights` directory. Call `load_weights` method to use them.
