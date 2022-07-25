import numpy as np
import scipy.stats as st

from .distribution import Distribution
from .symbolic_limit_state_functions import SymbolicLimitStateFunctions
from .event_vector import calculate_event_vector


def monte_carlo_simulation(distribution_dataset, correlation_xx, limit_state_functions, system_failure, simulation_number, simulation_tolerance):
    distribution_dataset = np.array(distribution_dataset)
    correlation_xx = np.array(correlation_xx)

    variable_number = len(distribution_dataset)
    component_number = len(limit_state_functions)
    event_vector = calculate_event_vector(system_failure, component_number)

    variables = Distribution(distribution_dataset, correlation_xx)
    limit_state_functions_object = SymbolicLimitStateFunctions(variable_number, limit_state_functions)

    index_function_value = np.array([])
    failure_probability = np.array([])
    standard_deviation_pf = np.array([])
    coefficient_of_variance_pf = np.array([])
    compute_vector = np.zeros(component_number)

    for i in range(component_number):
        compute_vector[i] = 2**i

    for iteration in range(1, simulation_number+1):

        u_sample = np.random.normal(size=variable_number)
        x_sample = variables.u2x(u_sample)

        if iteration == 1:
            x_stack = x_sample.copy()
        else:
            x_stack = np.append(x_stack, x_sample)

        sample_failure_index = int(compute_vector @ (limit_state_functions_object.g(x_sample) > 0).T)
        if event_vector[sample_failure_index] == 1:
            index_function_value = np.append(index_function_value, 1)
        else:
            index_function_value = np.append(index_function_value, 0)

        failure_probability = np.append(failure_probability, np.sum(index_function_value)/iteration)
        standard_deviation_pf = np.append(standard_deviation_pf, np.sqrt(failure_probability[-1]*(1-failure_probability[-1])/iteration))
        if failure_probability[-1] == 0:
            coefficient_of_variance_pf = np.append(coefficient_of_variance_pf, np.inf)
        else:
            coefficient_of_variance_pf = np.append(coefficient_of_variance_pf, standard_deviation_pf[-1]/failure_probability[-1])

        if iteration % 500 == 0:
            print('MCS Iteration #{0} complete'.format(iteration))

        if (coefficient_of_variance_pf[-1] < simulation_tolerance) & (sum(index_function_value) != iteration):
            break

    mcs_results = {
        'index': index_function_value,
        'beta': -st.norm.isf(1-failure_probability[-1]),
        'num_sim': iteration,
        'Pf': failure_probability[-1],
        'Pf_stack': failure_probability,
        'cov_Pf': coefficient_of_variance_pf[-1],
        'cov_Pf_stack': coefficient_of_variance_pf,
        'std_Pf_stack': standard_deviation_pf,
        'x_stack': x_stack,
        'event_vec': event_vector
    }

    return mcs_results
