import numpy as np
import matplotlib.pyplot as plt

from strap.first_order_reliability_method import first_order_reliability_method
from strap.second_order_reliability_method import second_order_reliability_method
from strap.system import matrix_based_system_reliability
from strap.monte_carlo_simulation import monte_carlo_simulation
from strap.importance_sampling import importance_sampling

print('--------------------------------------------------------')
print('|                    Welcome to STRAP                  |')
print('|           Which option do you want to use?           |')
print('--------------------------------------------------------')
print('')
print('0: Exit')
print('1: Help')
print('2: FORM Analysis')
print('3: SORM Analysis (Curvature-fitting)')
print('4: System Analysis (Matrix-based System Reliability)')
print('5: Monte Carlo Simulation Analysis')
print('6: Importance Sampling Simulation Analysis')
print('')
option = input('Choose option : ')
np.set_printoptions(precision=6, floatmode='fixed')
if option == '0':
    print('')
    print('--------------------------------------------------------')
    print('Option 0: Exit')
    print('Thank you. Bye.')

elif option == '1':
    print('')
    print('--------------------------------------------------------')
    print('Option 1: Help')
    helptxt = open('Help.txt', 'r')
    txt = helptxt.read()
    print(txt)
    helptxt.close()
elif option == '2':
    print('')
    print('--------------------------------------------------------')
    print('Option 2: FORM Analysis')
    if (len(limit_state_functions) > 1):
        print('Too many limit-state functions.')
    else:
        print('Computing...')
        formresults = first_order_reliability_method(distribution_dataset, correlation_xx, limit_state_functions, tolerance, max_iteration, compute_type)
        print('Done!')
        print('')
        print('---------------')
        print('| FORM RESULT |')
        print('---------------')
        print('')
        print('Number of iteration: {0}'.format(formresults['iteration']))
        print('Reliability index beta1: {:.6g}'.format(formresults['beta1'][0, 0]))
        print('Failure probability Pf1: {:.6g}'.format(formresults['Pf1'][0, 0]))
        print('')
        print('SENSITIVITIES OF THE RELIABILITY INDEX WITH RESPECT TO DISTRIBUTION PARAMETERS')
        print('-------------------------------------------------------------------------')
        print((formresults['beta_sensi_thetaf']))
        print('-------------------------------------------------------------------------')
        print('')
        print('SENSITIVITIES of THE FAILURE PROBABILITY WITH RESPECT TO DISTRIBUTION PARAMETERS')
        print('---------------------------------------------------------------------------')
        print((formresults['Pf_sensi_thetaf']))
        print('---------------------------------------------------------------------------')
        print('')
        print('')
        print('formresults[\'iteration\']         = Number of iterations')
        print('formresults[\'beta1\']             = Reliability index beta from FORM analysis')
        print('formresults[\'Pf1\']               = Failure probability pf1')
        print('formresults[\'u_design\']          = Design point u_star')
        print('formresults[\'x_design\']          = Design point in original space')
        print('formresults[\'u_stack\']           = Stack of points in u space')
        print('formresults[\'x_stack\']           = Stack of points in original space')
        print('formresults[\'alpha\']             = Alpha vector')
        print('formresults[\'imptg\']             = Importance vector gamma')
        print('formresults[\'g_stack\']           = Stack of the limit state function during search')
        print('formresults[\'beta_sensi_thetaf\'] = Beta sensitivities with respect to distribution parameters')
        print(
            'formresults[\'Pf_sensi_thetaf\']   = Probability of failure sensitivities with respect to distribution parameters')
        output = open("output_file.txt", 'a')
        output.write('---------------------------------------FORM RESULT---------------------------------------\n')
        for key, value in formresults.items():
            output.write(str(key) + " : \n")
            output.write(str(value) + '\n\n')
        output.close()

elif option == '3':
    print('')
    print('--------------------------------------------------------')
    print('Option 3: SORM Analysis')
    if (len(limit_state_functions) > 1):
        print('Too many limit-state functions.')
    else:
        print('Computing...')
        formresults, sormresults = second_order_reliability_method(distribution_dataset, correlation_xx, limit_state_functions, tolerance, max_iteration, compute_type)
        print('Done!')
        print('')
        print('---------------')
        print('| FORM RESULT |')
        print('---------------')
        print('')
        print('Number of iteration: {0}'.format(formresults['iteration']))
        print('Reliability index beta1: {:.6g}'.format(formresults['beta1'][0]))
        print('Failure probability Pf1: {:.6g}'.format(formresults['Pf1'][0]))
        print('')
        print('---------------')
        print('| SORM RESULT |')
        print('---------------')
        print('')
        print('# Main curvature in (n-1) * (n-1) space #')
        print(sormresults['kappa'])
        print('')
        print('')
        print('# Breitung formula #')
        print('Reliability_index beta2: {:.6g}'.format(sormresults['beta2_breitung'][0][0]))
        print('Failure probability Pf2: {:.6g}'.format(sormresults['Pf2_breitung'][0][0]))
        print('')
        print('')
        print('# Improved Breitung #')
        print('Reliability_index beta2: {:.6g}'.format(sormresults['beta2_ibreitung'][0][0]))
        print('Failure probability Pf2: {:.6g}'.format(sormresults['Pf2_ibreitung'][0][0]))
        print('')
        print('')
        print('formresults[\'iteration\']         = Number of iterations')
        print('formresults[\'beta1\']             = Reliability index beta from FORM analysis')
        print('formresults[\'Pf1\']               = Failure probability pf1')
        print('formresults[\'u_design\']          = Design point u_star')
        print('formresults[\'x_design\']          = Design point in original space')
        print('formresults[\'u_stack\']           = stack of points in u space')
        print('formresults[\'x_stack\']           = stack of points in original space')
        print('formresults[\'alpha\']             = Alpha vector')
        print('formresults[\'imptg\']             = Importance vector gamma')
        print('formresults[\'g_stack\']           = Recorded values of the limit state function during search')
        print('formresults[\'beta_sensi_thetaf\'] = Beta sensitivities with respect to distribution parameters')
        print(
            'formresults[\'Pf_sensi_thetaf\']   = Probability of failure sensitivities with respect to distribution parameters')
        print('')
        print('')
        print(
            'sormresults[\'beta2_breitung\']    = Reliability index beta2 SORM curvature fitting analysis and Breitung formula')
        print(
            'sormresults[\'Pf2_breitung\']      = Failure probability pf2 SORM curvature fitting analysis and Breitung formula')
        print(
            'sormresults[\'beta2_ibreitung\']   = Reliability index beta2 SORM curvature fitting analysis and Improved Breitung formula')
        print(
            'sormresults[\'Pf2_ibreitung\']     = Failure probability pf2 SORM curvature fitting analysis and Improved Breitung formula')
        print('sormresults[\'hess_G\']            = Hessian of G evaluated at the design point')
        print('sormresults[\'kappa\']             = Curvatures in the (n-1)(n-1) space')
        print('sormresults[\'eig_vectors\']       = Matrix of eigenvectors')
        print('sormresults[\'R1\']                = Rotation matrix')
        output = open("output_file.txt", 'a')
        output.write('---------------------------------------FORM RESULT---------------------------------------\n')
        for key, value in formresults.items():
            output.write(str(key) + " : \n")
            output.write(str(value) + '\n\n')
        output.write('---------------------------------------SORM RESULT---------------------------------------\n')
        for key, value in sormresults.items():
            output.write(str(key) + " : \n")
            output.write(str(value) + '\n\n')
        output.close()

elif option == '4':
    print('')
    print('--------------------------------------------------------')
    print('Option 4: System Analysis')
    print('Computing...')
    systemresults = matrix_based_system_reliability(distribution_dataset, correlation_xx, limit_state_functions, system_failure, tolerance, max_iteration, compute_type)
    print('Done!')
    print('')
    print('-----------------')
    print('| SYSTEM RESULT |')
    print('-----------------')
    print('')
    print('Event vector: {0}'.format(systemresults['event_vec']))
    print('DS class correlation coefficients: {0}'.format(systemresults['r']))
    print('Reliability index beta beta_sys: {:.6g}'.format(systemresults['beta_sys'][0]))
    print('Failure probability Pf_sys: {:.6g}'.format(systemresults['Pf_sys'][0]))
    print('')
    print('')
    print('systemresults[\'iteration_i\'] = Number of iterations of each component')
    print('systemresults[\'u_desgin_i\'] = Design point u_star of each component')
    print('systemresults[\'x_desgin_i\'] = Design point in original space of each component')
    print('systemresults[\'beta_i\'] = Reliability index beta from FORM analysis of each component')
    print('systemresults[\'Pf_i\'] = Failure probability from FORM analysis of each component')
    print('systemresults[\'alpha_i\'] = Alpha vector of each component')
    print('')
    print('systemresults[\'event_vec\'] = Event vector of system')
    print('systemresults[\'R_comp\'] = Correlation matrix of components')
    print('systemresults[\'r\'] = DS class correlation coefficients')
    print('systemresults[\'R_DS\'] = DS class correlation matrix approximated by least square method')
    print('systemresults[\'P_tilda\'] = Failure probability of each basic MECE event')
    print('systemresults[\'Pf_sys\'] = Failure probability of system')
    print('systemresults[\'beta_sys\'] = Reliability index beta of system')
    output = open("output_file.txt", 'a')
    output.write('--------------------------------------SYSTEM RESULT--------------------------------------\n')
    for key, value in systemresults.items():
        output.write(str(key) + " : \n")
        output.write(str(value) + '\n\n')
    output.close()

elif option == '5':
    print('')
    print('--------------------------------------------------------')
    print('Option 5: Monte Carlo Simulation Analysis')
    print('Computing...')
    print('--------------------------------------------------------')
    mcsresults = monte_carlo_simulation(distribution_dataset, correlation_xx, limit_state_functions, system_failure, simulation_number, simulation_tolerance)
    print('--------------------------------------------------------')
    print('Done!')
    print('')
    print('--------------')
    print('| MCS RESULT |')
    print('--------------')
    print('')
    print('Number of simulations: {0}'.format(mcsresults['num_sim']))
    print('Reliability index beta beta: {:.6g}'.format(mcsresults['beta']))
    print('Failure probability Pf: {:.6g}'.format(mcsresults['Pf']))
    print('Coefficient of variation of probability of failure: {:.6g}'.format(mcsresults['cov_Pf']))
    print('')
    print('')
    print('mcsresults[\'event_vec\'] = Event vector of system')
    print('mcsresults[\'num_sim\'] = Number of simulations')
    print('mcsresults[\'beta\'] = Reliability index beta from Monte Carlo Simulation')
    print('mcsresults[\'Pf\'] = Failure probability from Monte Carlo Simulation')
    print('mcsresults[\'cov_Pf\'] = Coefficient of variation of failure probability from Monte Carlo Simulation')
    print('')
    print('mcsresults[\'x_stack\'] = Stack of simulated points')
    print('mcsresults[\'Pf_stack\'] = Stack of failure probability of each simulation')
    print('mcsresults[\'std_Pf_stack\'] = Stack of standard deviation of failure probability of each simulation')
    print('mcsresults[\'cov_Pf_stack\'] = Stack of coefficient of variation of failure probability of each simulation')
    output = open("output_file.txt", 'a')
    output.write('----------------------------------------MCS RESULT----------------------------------------\n')
    for key, value in mcsresults.items():
        output.write(str(key) + " : \n")
        output.write(str(value) + '\n\n')
    output.close()
    plt.plot(mcsresults['Pf_stack'], 'dodgerblue', label='Pf by MCS')
    plt.plot(mcsresults['Pf_stack'] + 2 * mcsresults['std_Pf_stack'], 'gray', label='+/- 2std by MCS (95%)')
    plt.plot(mcsresults['Pf_stack'] - 2 * mcsresults['std_Pf_stack'], 'gray')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Pf')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.show()

elif option == '6':
    print('')
    print('--------------------------------------------------------')
    print('Option 6: Importance Sampling Simulation Analysis')
    print('Computing...')
    print('--------------------------------------------------------')
    isresults = importance_sampling(distribution_dataset, correlation_xx, limit_state_functions, system_failure, compute_type, tolerance, max_iteration, simulation_number, simulation_tolerance, sampling_covariance, m)
    print('--------------------------------------------------------')
    print('Done!')
    print('')
    print('-------------')
    print('| IS RESULT |')
    print('-------------')
    print('')
    print('Number of simulations: {0}'.format(isresults['num_sim']))
    print('Reliability index beta beta: {:.6g}'.format(isresults['beta']))
    print('Failure probability Pf: {:.6g}'.format(isresults['Pf']))
    print('Coefficient of variation of probability of failure: {:.6g}'.format(isresults['cov_Pf']))
    print('')
    print('')
    print('isresults[\'event_vec\'] = Event vector of system')
    print('isresults[\'num_sim\'] = Number of simulations')
    print('isresults[\'beta\'] = Reliability index beta from Importance Sampling Simulation')
    print('isresults[\'Pf\'] = Failure probability from Importance Sampling Simulation')
    print('isresults[\'cov_Pf\'] = Coefficient of variation of failure probability from Importance Sampling Simulation')
    print('')
    print('isresults[\'x_stack\'] = Stack of simulated points')
    print('isresults[\'Pf_stack\'] = Stack of failure probability of each simulation')
    print('isresults[\'std_Pf_stack\'] = Stack of standard deviation of failure probability of each simulation')
    print('isresults[\'cov_Pf_stack\'] = Stack of coefficient of variation of failure probability of each simulation')
    output = open("output_file.txt", 'a')
    output.write('----------------------------------------IS RESULT----------------------------------------\n')
    for key, value in isresults.items():
        output.write(str(key) + " : \n")
        output.write(str(value) + '\n\n')
    output.close()
    plt.plot(isresults['Pf_stack'], 'dodgerblue', label='Pf by IS')
    plt.plot(isresults['Pf_stack'] + 2 * isresults['std_Pf_stack'], 'gray', label='+/- 2std by IS (95%)')
    plt.plot(isresults['Pf_stack'] - 2 * isresults['std_Pf_stack'], 'gray')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Pf')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.show()
else:
    print('Unexpected option.')
    print('Bye.')