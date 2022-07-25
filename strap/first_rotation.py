import numpy as np


def first_rotation(alpha):
    variable_number = len(alpha)
    rotation_matrix = np.eye(variable_number)
    rotation_matrix[variable_number-1] = alpha

    for i in range(variable_number):
        k = variable_number-i-1
        x_k = rotation_matrix[k]
        parameter = rotation_matrix[0:k] @ x_k.T/np.sum(x_k**2)

        for j in range(k):
            rotation_matrix[j] -= parameter[j]*x_k
            if j == k-1:
                rotation_matrix[j] /= np.sqrt(np.sum(rotation_matrix[j]**2))

    return rotation_matrix
