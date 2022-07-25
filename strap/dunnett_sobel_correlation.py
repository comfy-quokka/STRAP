import numpy as np
from scipy.optimize import minimize


def dunnett_sobel_correlation(correlation_component):
    component_number = len(correlation_component)

    coefficient_initial = 0.5 * np.ones(component_number)
    bound = [[-1, 1]] * component_number

    least_square = minimize(dunnett_sobel_correlation_cost, coefficient_initial, method='SLSQP', bounds=bound, args=correlation_component, tol=1e-6)
    correlation_coefficient = least_square.x.reshape(component_number)
    correlation_matrix = np.diag(np.ones(component_number)) + correlation_coefficient.reshape(-1, 1) @ correlation_coefficient.reshape(-1, 1).T - np.diag(correlation_coefficient ** 2)

    return correlation_coefficient, correlation_matrix


def dunnett_sobel_correlation_cost(x, correlation_component):
    component_number = len(correlation_component)
    r = x.reshape(1, component_number)

    cost_matrix = correlation_component - r.T @ r + np.diag(x**2) - np.diag(np.ones(component_number))
    cost = np.sum(cost_matrix ** 2)

    return cost
