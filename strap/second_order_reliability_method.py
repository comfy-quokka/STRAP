import numpy as np
from scipy.stats import norm

from .distribution import Distribution
from .symbolic_limit_state_functions import SymbolicLimitStateFunctions
from .hessian import hessian
from .first_rotation import first_rotation


def second_order_reliability_method(distribution_dataset, correlation_xx, limit_state_functions, tolerance, max_iteration, compute_type):
    distribution_dataset = np.array(distribution_dataset)
    correlation_xx = np.array(correlation_xx)
    variables = Distribution(distribution_dataset, correlation_xx)
    limit_state_function_object = SymbolicLimitStateFunctions(variables.variable_number, limit_state_functions)

    # starting point
    x_prev = variables.starting_point.copy()
    x_stack = x_prev.copy()
    u_prev = variables.x2u(x_prev)
    u_stack = u_prev.copy()
    G0 = limit_state_function_object.g(x_prev)

    g_stack = np.array([])
    convergence = 'fail'

    for iteration in range(1, max_iteration+1):
        G_ui = limit_state_function_object.g(x_prev)
        g_stack = np.append(g_stack, G_ui)
        jacobian_xu = variables.jacobian_xu(x_prev)
        gradient_G_ui = limit_state_function_object.gradient_g(x_prev) @ jacobian_xu
        alpha_i = -gradient_G_ui/np.linalg.norm(gradient_G_ui)

        if abs(G_ui/G0) < tolerance[0] and np.linalg.norm(u_prev - alpha_i@u_prev*alpha_i.T) < tolerance[1]:
            convergence = 'success'
            break

        step_size = (alpha_i@u_prev + G_ui/np.linalg.norm(gradient_G_ui))*alpha_i.T - u_prev

        if compute_type == 'HLRF':
            u_prev = u_prev + step_size
            x_prev = variables.u2x(u_prev)
            u_stack = np.append(u_stack, u_prev, axis=1)
            x_stack = np.append(x_stack, x_prev, axis=1)
        else:
            c = (int(np.linalg.norm(u_prev) / np.linalg.norm(gradient_G_ui) / 10) + 1) * 10
            merit_prev = 0.5 * sum(u_prev ** 2) + c * abs(G_ui)  # merit function
            while True:
                u_post = u_prev + step_size
                x_post = variables.u2x(u_post)
                G_ui_post = limit_state_function_object.g(x_post)
                merit_post = 0.5*sum(u_post**2)+c*abs(G_ui_post)
                if merit_post < merit_prev:
                    u_stack = np.append(u_stack, u_post, axis=1)
                    x_stack = np.append(x_stack, x_post, axis=1)
                    u_prev = u_post
                    x_prev = x_post
                    break
                step_size = step_size/2

    u_design = u_prev
    x_design = x_prev
    alpha = alpha_i
    beta1 = alpha @ u_design
    jacobian_ux = np.linalg.inv(jacobian_xu)
    D_hat = np.sqrt(np.diag(np.diag(jacobian_xu @ jacobian_xu.T)))
    beta_sensitivity_theta_f, pf_sensitivity_theta_f = variables.sensitivity_theta_f(alpha, beta1, x_prev)
    form_results = {
        'convergence': convergence,
        'iteration': iteration,
        'beta1': beta1,
        'Pf1': norm.cdf(-beta1),
        'u_design': u_design,
        'x_design': x_design,
        'u_stack': u_stack,
        'x_stack': x_stack,
        'alpha': alpha,
        'imptg': (alpha @ jacobian_ux @ D_hat / np.linalg.norm(alpha @ jacobian_ux @ D_hat))[0],
        'g_stack': g_stack,
        'beta_sensi_thetaf': beta_sensitivity_theta_f,
        'Pf_sensi_thetaf': pf_sensitivity_theta_f
    }

    hessian_matrix = hessian(variables, limit_state_function_object, u_design, x_design)
    first_rotation_matrix = first_rotation(alpha[0])
    A11 = (first_rotation_matrix @ hessian_matrix @ first_rotation_matrix.T / np.sqrt(np.sum(gradient_G_ui**2)))[:-1, :-1]

    curvature, eigenvectors = np.linalg.eig(A11)

    psi_beta = norm.pdf(beta1) / norm.cdf(-beta1)
    Pf2_breitung = (norm.cdf(-beta1) * np.prod(1 / np.sqrt(1 + beta1 * curvature)))
    beta2_breitung = (-norm.isf(1 - Pf2_breitung))
    Pf2_ibreitung = (norm.cdf(-beta1) * np.prod(1 / np.sqrt(1 + psi_beta * curvature)))
    beta2_ibreitung = (-norm.isf(1 - Pf2_ibreitung))

    sorm_results = {
        "kappa": curvature,
        "eig_vectors": eigenvectors,
        "beta2_breitung": beta2_breitung,
        "Pf2_breitung": Pf2_breitung,
        "beta2_ibreitung": beta2_ibreitung,
        "Pf2_ibreitung": Pf2_ibreitung,
        "R1": first_rotation_matrix,
        "hess_G": hessian_matrix
    }

    return form_results, sorm_results
