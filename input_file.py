import numpy as np

################################################## Input data ##################################################
# distribution_dataset = [[distribution type, mean, standard deviation, starting point], -> variable x[1]
#                         [distribution type, mean, standard deviation, starting point], -> variable x[2]
#                          ...]
distribution_dataset = [[1, 30, 9, 30],
                        [1, 30, 9, 30],
                        [1, 30, 3, 30],
                        [1, 45, 4.5, 45],
                        [1, 25, 2.5, 25]]

# Correlation_xx = correlation matrix
correlation_xx = np.ones((5,5))*0.2 + 0.8*np.eye(5)

# limit_state_funtions = ['Limit-state function #1',
#                           'Limit-state function #2'],
#                           ... ]
limit_state_functions = ['2*x[3]+x[4]-2*x[1]',
                         'x[4]+2*x[5]-2*x[1]',
                         '-2*x[1]-2*x[2]+2*x[3]+2*x[4]+2*x[5]']

# tolerance = [e1, e2]
# e1 = tolerance on how close design point is to limit-state surface
# e2 = tolerance on how accurately the gradient points towards the origin
# (Do not need for Monte Carlo Simulation)
tolerance = [1e-3, 1e-3]

# compute_type = 'HLRF'  for HL-RF algorithm
#              = 'iHLRF' for Improved HL-RF algorithm
compute_type = 'HLRF'

# MaxIteration = maximum iteration for FORM analysis
max_iteration = 100

# system_failure = experssion of system failure
#                = [[cutset #1 of system with component number],
#                   [cutset #2 of system with component number], ...]
# (Do not need for FORM, SORM)
system_failure = [[1],[2],[3]]

# simulation_number = number of simulation5
# (Do not need for FORM, SORM, System Reliability)
simulation_number = 5000

# simulation_tolerance = tolerance on how accurately simulate
# (Do not need for FORM, SORM, System Reliability)
simulation_tolerance = 1e-2

# sampling_covariance = sampling covariance matrix for Importance Sampling
# (Do not need for FORM, SORM, System Reliability, Monte Carlo Simulation)
sampling_covariance = np.eye(5)

# m = power of beta when compute weight of each component
# (Do not need for FORM, SORM, System Reliability, Monte Carlo Simulation)
m = 1

################################################################################################################
exec(open('operator.py').read())
