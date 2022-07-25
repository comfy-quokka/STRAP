from sympy import *
import numpy as np


class SymbolicLimitStateFunctions:

    def __init__(self, variable_number, limit_state_functions):
        symbol_code = ''
        for i in range(variable_number-1):
            symbol_code += 'x' + str(i+1) + ','
        symbol_code += 'x' + str(variable_number)
        x = [0]*(variable_number+1)
        x[1:] = symbols(symbol_code)

        symbolic_functions = []
        for component in limit_state_functions:
            symbolic_functions.append(eval(component))

        self.variable_number = variable_number
        self.component_number = len(limit_state_functions)
        self.symbol_code = symbol_code
        self.symbolic_limit_state_functions = symbolic_functions

    def g(self, x_input):
        x = [0]*(self.variable_number+1)
        x[1:] = symbols(self.symbol_code)
        g_input = [0]*self.variable_number
        g = np.zeros(self.component_number)

        # make tuple to put in subs function and compute g(x)
        for i in range(self.variable_number):
            g_input[i] = (x[i+1], x_input[i][0])
        for i in range(self.component_number):
            g[i] = float(self.symbolic_limit_state_functions[i].subs(g_input))

        return g

    def gradient_g(self, x_input):
        x = [0]*(self.variable_number+1)
        x[1:] = symbols(self.symbol_code)
        gradient_g_input = [0]*self.variable_number
        gradient_g = np.zeros((self.component_number, self.variable_number))

        # make tuple to put in subs function and compute gradient_g(x)
        for i in range(self.variable_number):
            gradient_g_input[i] = (x[i+1], x_input[i][0])
        for i in range(self.component_number):
            for k in range(self.variable_number):
                gradient_g_k = self.symbolic_limit_state_functions[i].diff(x[k+1])
                gradient_g[i][k] = float(gradient_g_k.subs(gradient_g_input))

        return gradient_g
