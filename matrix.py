def map(func, matrix):
    x, y = matrix.shape
    for i in range(x):
        for j in range(y):
            matrix[i, j] = func(matrix[i, j])
