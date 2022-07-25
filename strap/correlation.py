import numpy as np


# calculate correlation matrix of z (Nataf distribution)
def nataf_correlation(variable_number, distribution_type, mean, standard_deviation, correlation_xx):
    correlation_zz = np.zeros((variable_number, variable_number))
    delta = np.zeros(variable_number)

    # compute c.o.v.
    for i in range(variable_number):
        if mean[i] != 0:
            delta[i] = standard_deviation[i]/mean[i]

    # compute correlation
    for i, distribution_i in enumerate(distribution_type):
        for j, distribution_j in enumerate(distribution_type):
            # diagonal elements are 1
            if i == j:
                correlation_zz[i, j] = 1
            else:
                correlation_zz[i, j] = correlation_xx[i, j] * nataf_f(distribution_i, distribution_j, delta[i], delta[j], correlation_xx[i, j])

    return correlation_zz


def nataf_f(distribution_i, distribution_j, delta_i, delta_j, correlation):
    f = 0

    # compute only for i <= j
    if distribution_i > distribution_j:
        distribution_i, distribution_j = distribution_j, distribution_i
        delta_i, delta_j = delta_j, delta_i

    if distribution_i == 1:  # Noraml
        if distribution_j == 1:  # Noraml
            f = 1
        elif distribution_j == 2:  # Lognormal
            f = delta_j / np.sqrt(np.log(1 + delta_j ** 2))
        elif distribution_j == 3:  # Gamma
            f = 1.001 - 0.007 * delta_j + 0.118 * delta_j ** 2
        elif distribution_j == 4:  # Exponential
            f = 1.107
        elif distribution_j == 5:  # Rayleigh
            f = 1.014
        elif distribution_j == 6:  # Uniform
            f = 1.023

    elif distribution_i == 2:  # Lognormal
        if distribution_j == 2:  # Lognormal
            f = np.log(1 + correlation * delta_i * delta_j) / (correlation * np.sqrt(np.log(1 + delta_i ** 2) * np.log(1 + delta_j ** 2)))
        elif distribution_j == 3:  # Gamma
            f = 1.001 + 0.033 * correlation + 0.004 * delta_i - 0.016 * delta_j + 0.002 * correlation ** 2 + 0.223 * delta_i ** 2 + 0.13 * delta_j ** 2 - 0.104 * correlation * delta_i + 0.029 * delta_i * delta_j - 0.119 * correlation * delta_j
        elif distribution_j == 4:  # Exponential
            f = 1.098 + 0.003 * correlation + 0.019 * delta_i + 0.025 * correlation ** 2 + 0.303 * delta_i ** 2 - 0.437 * correlation * delta_i
        elif distribution_j == 5:  # Rayleigh
            f = 1.011 + 0.001 * correlation + 0.014 * delta_i + 0.004 * correlation ** 2 + 0.231 * delta_i ** 2 - 0.130 * correlation * delta_i
        elif distribution_j == 6:  # Uniform
            f = 1.019 + 0.014 * delta_i + 0.01 * correlation ** 2 + 0.249 * delta_i ** 2

    elif distribution_i == 3:  # Gamma
        if distribution_j == 3:  # Gamma
            f = 1.002 + 0.022 * correlation - 0.012 * (delta_i + delta_j) + 0.001 * correlation ** 2 + 0.125 * (delta_i ** 2 + delta_j ** 2) - 0.077 * correlation * (delta_i + delta_j) + 0.014 * delta_i * delta_j
        elif distribution_j == 4:  # Exponential
            f = 1.104 + 0.003 * correlation - 0.008 * delta_i + 0.014 * correlation ** 2 + 0.173 * delta_i ** 2 - 0.296 * correlation * delta_i
        elif distribution_j == 5:  # Rayleigh
            f = 1.104 + 0.001 * correlation - 0.007 * delta_i + 0.002 * correlation ** 2 + 0.126 * delta_i ** 2 - 0.09 * correlation * delta_i
        elif distribution_j == 6:  # Uniform
            f = 1.023 - 0.007 * delta_i + 0.002 * correlation ** 2 + 0.127 * delta_i ** 2

    elif distribution_i == 4:  # Exponential
        if distribution_j == 4:  # Exponential
            f = 1.229 - 0.367 * correlation + 0.153 * correlation ** 2
        elif distribution_j == 5:  # Rayleigh
            f = 1.123 - 0.1 * correlation + 0.021 * correlation ** 2
        elif distribution_j == 6:  # Uniform
            f = 1.133 + 0.029 * correlation ** 2

    elif distribution_i == 5:  # Rayleigh
        if distribution_j == 5:  # Rayleigh
            f = 1.028 - 0.029 * correlation
        elif distribution_j == 6:  # Uniform
            f = 1.038 - 0.008 * correlation ** 2

    elif distribution_i == 6:  # Uniform
        if distribution_j == 6:  # Uniform
            f = 1.047 - 0.047 * correlation ** 2

    return f
