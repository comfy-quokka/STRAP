import numpy as np


def calculate_event_vector(system_failure, component_number):
    event_set = np.array([[1, 0]], dtype='int')

    for i in range(1, component_number):
        upper = np.vstack((event_set, np.ones((1, 2 ** i), dtype='int')))
        lower = np.vstack((event_set, np.zeros((1, 2 ** i), dtype='int')))
        event_set = np.hstack((upper, lower))

    event_vector = np.zeros(2**component_number, dtype='int')
    for cutset in system_failure:
        cutset_vector = [True]*(2**component_number)
        for c in cutset:
            if c > 0:
                tmp = (event_set[c-1, :] == 1)
            else:
                tmp = (event_set[-c-1, :] == 0)
            cutset_vector = cutset_vector & tmp
        event_vector = cutset_vector | event_vector

    return event_vector
