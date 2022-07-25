import numpy as np
import scipy.stats as st

from .distribution import Distribution
from .symbolic_limit_state_functions import SymbolicLimitStateFunctions
from .event_vector import calculate_event_vector
from .first_order_reliability_method import first_order_reliability_method


def importance_sampling(distribution_dataset, correlation_xx, limit_state_functions, system_failure, compute_type, tolerance, max_iteration, simulation_number, simulation_tolerance, sampling_covariance, m):
    distribution_dataset = np.array(distribution_dataset)
    correlation_xx = np.array(correlation_xx)
    sampling_covariance = np.array(sampling_covariance)

    variable_number = len(distribution_dataset)
    component_number = len(limit_state_functions)
    event_vector = calculate_event_vector(system_failure, component_number)

    variables = Distribution(distribution_dataset, correlation_xx)
    limit_state_functions_object = SymbolicLimitStateFunctions(variable_number, limit_state_functions)

    u_design_set = np.zeros((variable_number, component_number))
    beta_set = np.zeros(component_number)
    convergence_set = []

    for i in range(component_number):
        form_results_i = first_order_reliability_method(distribution_dataset, correlation_xx, [limit_state_functions[i]], tolerance, max_iteration, compute_type)
        u_design_set[:, i] = form_results_i['u_design'].reshape(-1)
        beta_set[i] = form_results_i['beta1']
        convergence_set.append(form_results_i['convergence'])

    weight = (beta_set ** -m).copy()
    weight = (weight / np.sum(weight)).reshape(component_number, 1)

    sampling_mean = (u_design_set @ weight).reshape(variable_number)

    x_stack = np.array([])
    index_function_value = np.array([])
    q = np.array([])
    failure_probability = np.array([])
    standard_deviation_pf = np.array([])
    coefficient_of_variance_pf = np.array([])
    compute_vector = np.zeros(component_number)

    for i in range(component_number):
        compute_vector[i] = 2**i

    for iteration in range(1, simulation_number + 1):
        u_sample = np.random.multivariate_normal(sampling_mean, sampling_covariance)
        f_sample = st.multivariate_normal.pdf(u_sample, np.zeros(variable_number), np.eye(variable_number))
        h_sample = st.multivariate_normal.pdf(u_sample, sampling_mean, sampling_covariance)
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

        q = np.append(q, index_function_value[-1]*f_sample/h_sample)
        failure_probability = np.append(failure_probability, np.sum(q)/iteration)
        standard_deviation_pf = np.append(standard_deviation_pf, np.sqrt(np.var(q)/iteration))
        if failure_probability[-1] == 0:
            coefficient_of_variance_pf = np.append(coefficient_of_variance_pf, np.inf)
        else:
            coefficient_of_variance_pf = np.append(coefficient_of_variance_pf, standard_deviation_pf[-1]/failure_probability[-1])

        if iteration % 500 == 0:
            print('IS Iteration #{0} complete'.format(iteration))

        if (coefficient_of_variance_pf[-1] < simulation_tolerance) & (sum(index_function_value) != iteration):
            break

    is_results = {
        'convergence': convergence_set,
        'Pf': failure_probability[-1],
        'index': index_function_value,
        'beta': -st.norm.isf(1-failure_probability[-1]),
        'num_sim': iteration,
        'Pf_stack': failure_probability,
        'std_Pf_stack': standard_deviation_pf,
        'cov_Pf': coefficient_of_variance_pf[-1],
        'cov_Pf_stack': coefficient_of_variance_pf,
        'x_stack': x_stack,
        'event_vec': event_vector
    }

    return is_results