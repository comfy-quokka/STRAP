import numpy as np


def hessian(variables, limit_state_function_object, u_design, x_design):
    du = 1/2000
    G0 = limit_state_function_object.g(x_design)
    G_front_collect = np.zeros((variables.variable_number, variables.variable_number))
    hessian = np.zeros((variables.variable_number, variables.variable_number))

    for i in range(variables.variable_number):
        u_back = u_design.copy()
        u_front = u_design.copy()
        u_back[i] -= du
        u_front[i] += du

        x_back = variables.u2x(u_back)
        x_front = variables.u2x(u_front)

        G_back = limit_state_function_object.g(x_back)
        G_front = limit_state_function_object.g(x_front)
        G_front_collect[i, i] = G_front

        hessian[i, i] = (G_front - 2*G0 + G_back)/du**2
        u_front_i = u_front.copy()

        for j in range(i):
            u_front_ij = u_front_i.copy()
            u_front_ij[j] += du
            x_front_ij = variables.u2x(u_front_ij)
            G_front_collect[i, j] = limit_state_function_object.g(x_front_ij)

    for i in range(variables.variable_number):
        for j in range(i):
            # non diagonal elemnets of hessian matrix
            hessian[i, j] = ((G_front_collect[i, j]-G_front_collect[j, j])-(G_front_collect[i, i]-G0))/du**2
            hessian[j, i] = hessian[i, j]

    return hessian
