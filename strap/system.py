import numpy as np
from scipy.stats import norm

from .first_order_reliability_method import first_order_reliability_method
from .event_vector import calculate_event_vector
from .dunnett_sobel_correlation import dunnett_sobel_correlation


def matrix_based_system_reliability(distribution_dataset, correlation_xx, limit_state_functions, system_failure, tolerance, max_iteration, compute_type):
    component_number = len(limit_state_functions)
    variable_number = len(distribution_dataset)

    event_vector = calculate_event_vector(system_failure, component_number)
    alpha_set = np.zeros((component_number, variable_number))
    u_design_set = np.zeros((variable_number, component_number))
    x_design_set = np.zeros((variable_number, component_number))
    iteration_set = np.zeros(component_number)
    beta_set = np.zeros(component_number)
    pf1_set = np.zeros(component_number)
    convergence_set = []
    form_results_set = []

    for i in range(component_number):
        form_results_i = first_order_reliability_method(distribution_dataset, correlation_xx, [limit_state_functions[i]], tolerance, max_iteration, compute_type)
        alpha_set[i, :] = form_results_i['alpha']
        u_design_set[:, i] = form_results_i['u_design'].reshape(-1)
        x_design_set[:, i] = form_results_i['x_design'].reshape(-1)
        iteration_set[i] = form_results_i['iteration']
        beta_set[i] = form_results_i['beta1']
        pf1_set[i] = form_results_i['Pf1']
        convergence_set.append(form_results_i['convergence'])
        form_results_set.append(form_results_i)

    correlation_component = alpha_set @ alpha_set.T
    correlation_coefficient, correlation_matrix = dunnett_sobel_correlation(correlation_component)

    summation_range = 10
    ds = 0.005
    summation_space = np.linspace(-summation_range, summation_range, int(summation_range/ds*2+1))
    p_tilda = np.zeros(2 ** component_number)

    for point in summation_space:
        probability = np.array([1])

        for index, coefficient_i in enumerate(correlation_coefficient):
            if coefficient_i**2 != 1:
                probability_component = norm.cdf((-beta_set[index] - coefficient_i*point) / np.sqrt(1 - coefficient_i**2))
            else:
                probability_component = int(coefficient_i*point < -beta_set[index])

            probability = np.append(probability * probability_component, probability * (1-probability_component))

        p_tilda += norm.pdf(point)*ds*probability

    p_tilda = p_tilda.reshape(-1, 1)
    pf = event_vector @ p_tilda
    beta_system = -norm.isf(1-pf)

    system_results = {
        'convergence': convergence_set,
        'beta_i': beta_set,
        'Pf_i': pf1_set,
        'alpha_i': alpha_set,
        'R_comp': correlation_component,
        'R_DS': correlation_matrix,
        'r': correlation_coefficient,
        'P_tilda': p_tilda,
        'Pf_sys': pf,
        'beta_sys': beta_system,
        'iteration_i': iteration_set,
        'x_design_i': x_design_set,
        'u_design_i': u_design_set,
        'event_vec': event_vector,
        'formresults': form_results_set
    }

    return system_results
