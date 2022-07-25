from .correlation import nataf_correlation

import numpy as np
import scipy.stats as st


class Distribution:

    def __init__(self, distribution_data, correlation_xx):
        self.variable_number = len(distribution_data)
        self.distribution_type = distribution_data[:, 0].reshape(self.variable_number, 1)
        self.mean = distribution_data[:, 1].reshape(self.variable_number, 1)
        self.standard_deviation = distribution_data[:, 2].reshape(self.variable_number, 1)
        self.starting_point = distribution_data[:, 3].reshape(self.variable_number, 1)
        self.correlation_xx = correlation_xx
        self.correlation_zz = nataf_correlation(self.variable_number, self.distribution_type, self.mean, self.standard_deviation, correlation_xx)
        self.cholesky_zz = np.linalg.cholesky(self.correlation_zz)

        # compute parameter
        self.name = ['']*self.variable_number
        self.par1 = np.zeros(self.variable_number)
        self.par2 = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            mean_i = self.mean[index]
            standard_deviation_i = self.standard_deviation[index]

            if distribution_type_i == 1:
                # Normal: mean. std
                self.name[index] = 'Normal'
                self.par1[index] = mean_i
                self.par2[index] = standard_deviation_i

            elif distribution_type_i == 2:
                # Lognormal: lambda, zeta
                self.name[index] = 'Lognormal'
                self.par1[index] = np.log(mean_i) - 0.5 * np.log(1 + (standard_deviation_i / mean_i) ** 2)
                self.par2[index] = np.sqrt(np.log(1 + (standard_deviation_i / mean_i) ** 2))

            elif distribution_type_i == 3:
                # Gamma: a, b
                self.name[index] = 'Gamma'
                self.par1[index] = mean_i / standard_deviation_i ** 2
                self.par2[index] = (mean_i / standard_deviation_i) ** 2

            elif distribution_type_i == 4:
                # Shifted Exponential: lambda, location
                self.name[index] = 'Shifted Exponential'
                self.par1[index] = 1/standard_deviation_i
                self.par2[index] = mean_i - standard_deviation_i

            elif distribution_type_i == 5:
                # Shifted Rayleigh: sigma, location
                self.name[index] = 'Shifted Rayleigh'
                self.par1[index] = standard_deviation_i / np.sqrt((4 - np.pi) / 2)
                self.par2[index] = mean_i - standard_deviation_i / np.sqrt((4 - np.pi) / np.pi)

            elif distribution_type_i == 6:
                # Uniform: a, b
                self.name[index] = 'Uniform'
                self.par1[index] = mean_i - np.sqrt(3) * standard_deviation_i
                self.par2[index] = mean_i + np.sqrt(3) * standard_deviation_i

            else:
                print('distribution type error')

    # probability density function
    def pdf(self, x):
        pdf = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                pdf[index] = st.norm.pdf(x[index], par1_i, par2_i)

            elif distribution_type_i == 2:
                pdf[index] = st.lognorm.pdf(x[index], par2_i, 0, np.exp(par1_i))

            elif distribution_type_i == 3:
                pdf[index] = st.gamma.pdf(x[index], par2_i, 0, 1 / par1_i)

            elif distribution_type_i == 4:
                pdf[index] = st.expon.pdf(x[index], par2_i, 1 / par1_i)

            elif distribution_type_i == 5:
                pdf[index] = st.rayleigh.pdf(x[index], par2_i, par1_i)

            elif distribution_type_i == 6:
                pdf[index] = st.uniform.pdf(x[index], par1_i, par2_i - par1_i)

        return pdf.reshape(self.variable_number, 1)

    # cumulative distribution function
    def cdf(self, x):
        cdf = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                cdf[index] = st.norm.cdf(x[index], par1_i, par2_i)

            elif distribution_type_i == 2:
                cdf[index] = st.lognorm.cdf(x[index], par2_i, 0, np.exp(par1_i))

            elif distribution_type_i == 3:
                cdf[index] = st.gamma.cdf(x[index], par2_i, 0, 1 / par1_i)

            elif distribution_type_i == 4:
                cdf[index] = st.expon.cdf(x[index], par2_i, 1 / par1_i)

            elif distribution_type_i == 5:
                cdf[index] = st.rayleigh.cdf(x[index], par2_i, par1_i)

            elif distribution_type_i == 6:
                cdf[index] = st.uniform.cdf(x[index], par1_i, par2_i - par1_i)

        return cdf.reshape(self.variable_number, 1)

    # inverse Cumulative distribution function
    def icdf(self, p):
        icdf = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                icdf[index] = st.norm.isf(1 - p[index], par1_i, par2_i)

            elif distribution_type_i == 2:
                icdf[index] = st.lognorm.isf(1 - p[index], par2_i, 0, np.exp(par1_i))

            elif distribution_type_i == 3:
                icdf[index] = st.gamma.isf(1 - p[index], par2_i, 0, 1 / par1_i)

            elif distribution_type_i == 4:
                icdf[index] = st.expon.isf(1 - p[index], par2_i, 1 / par1_i)

            elif distribution_type_i == 5:
                icdf[index] = st.rayleigh.isf(1 - p[index], par2_i, par1_i)

            elif distribution_type_i == 6:
                icdf[index] = st.uniform.isf(1 - p[index], par1_i, par2_i - par1_i)

        return icdf.reshape(self.variable_number, 1)

    # generate random sample
    def random(self):
        output = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                output[index] = st.norm.rvs(par1_i, par2_i)

            elif distribution_type_i == 2:
                output[index] = st.lognorm.rvs(par2_i, 0, np.exp(par1_i))

            elif distribution_type_i == 3:
                output[index] = st.gamma.rvs(par2_i, 0, 1 / par1_i)

            elif distribution_type_i == 4:
                output[index] = st.expon.rvs(par2_i, 1 / par1_i)

            elif distribution_type_i == 5:
                output[index] = st.rayleigh.rvs(par2_i, par1_i)

            elif distribution_type_i == 6:
                output[index] = st.uniform.rvs(par1_i, par2_i - par1_i)

        return output.reshape(self.variable_number, 1)

    # transform x to z
    def x2z(self, x):
        z = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                z[index] = (x[index] - par1_i) / par2_i

            elif distribution_type_i == 2:
                f = st.lognorm.cdf(x[index], par2_i, 0, np.exp(par1_i))
                z[index] = st.norm.isf(1 - f)

            elif distribution_type_i == 3:
                f = st.gamma.cdf(x[index], par2_i, 0, 1 / par1_i)
                z[index] = st.norm.isf(1 - f)

            elif distribution_type_i == 4:
                f = st.expon.cdf(x[index], par2_i, 1 / par1_i)
                z[index] = st.norm.isf(1 - f)

            elif distribution_type_i == 5:
                f = st.rayleigh.cdf(x[index], par2_i, par1_i)
                z[index] = st.norm.isf(1 - f)

            elif distribution_type_i == 6:
                f = st.uniform.cdf(x[index], par1_i, par2_i - par1_i)
                z[index] = st.norm.isf(1 - f)

        return z.reshape(self.variable_number, 1)

    # transform z to x
    def z2x(self, z):
        x = np.zeros(self.variable_number)

        for index, distribution_type_i in enumerate(self.distribution_type):
            par1_i = self.par1[index]
            par2_i = self.par2[index]

            if distribution_type_i == 1:
                x[index] = par1_i + par2_i * z[index]

            elif distribution_type_i == 2:
                f = st.norm.cdf(z[index])
                x[index] = st.lognorm.isf(1 - f, par2_i, 0, np.exp(par1_i))

            elif distribution_type_i == 3:
                f = st.norm.cdf(z[index])
                x[index] = st.gamma.isf(1 - f, par2_i, 0, 1 / par1_i)

            elif distribution_type_i == 4:
                f = st.norm.cdf(z[index])
                x[index] = st.expon.isf(1 - f, par2_i, 1 / par1_i)

            elif distribution_type_i == 5:
                f = st.norm.cdf(z[index])
                x[index] = st.rayleigh.isf(1 - f, par2_i, par1_i)

            elif distribution_type_i == 6:
                f = st.norm.cdf(z[index])
                x[index] = st.uniform.isf(1 - f, par1_i, par2_i - par1_i)

        return x.reshape(self.variable_number, 1)

    def x2u(self, x_tmp):
        z_tmp = self.x2z(x_tmp)

        if np.array_equal(self.correlation_xx, np.eye(self.variable_number)):
            u_tmp = z_tmp
        else:
            u_tmp = np.linalg.inv(self.cholesky_zz) @ z_tmp

        return u_tmp.reshape(self.variable_number, 1)

    def u2x(self, u_tmp):
        if np.array_equal(self.correlation_xx, np.eye(self.variable_number)):
            z_tmp = u_tmp
        else:
            z_tmp = self.cholesky_zz @ u_tmp

        x_tmp = self.z2x(z_tmp)

        return x_tmp.reshape(self.variable_number, 1)

    def jacobian_xu(self, x_input):
        diagonal_std = np.diag(self.standard_deviation.reshape(self.variable_number))

        if np.array_equal(self.distribution_type, np.ones((self.variable_number, 1))):
            jacobian_matrix = diagonal_std @ self.cholesky_zz
        else:
            jacobian_xz = np.diag((st.norm.pdf(self.x2z(x_input))/self.pdf(x_input)).reshape(self.variable_number))

            if np.array_equal(self.correlation_xx, np.eye(self.variable_number)):
                jacobian_matrix = jacobian_xz
            else:
                jacobian_matrix = jacobian_xz @ self.cholesky_zz

        return jacobian_matrix

    def sensitivity_theta_f(self, alpha, beta, x_input):
        Jzt = np.zeros((self.variable_number, 4))
        beta_sensitivity_theta_f = np.zeros((self.variable_number, 4))
        delta = 1/2000

        for index, distribution_type_i in enumerate(self.distribution_type):
            mean = self.mean[index][0]
            standard_deviation = self.standard_deviation[index][0]
            par1 = self.par1[index]
            par2 = self.par2[index]
            x = x_input[index]

            if distribution_type_i == 1:  # Normal
                dzdm = -1 / standard_deviation
                dzds = -(x - mean) / standard_deviation ** 2
                dzdp1 = dzdm
                dzdp2 = dzds

            elif distribution_type_i == 2:  # Lognormal
                delta = standard_deviation / mean
                dzdp1 = -1 / par2
                dzdp2 = -(np.log(x) - par1) / par2 ** 2
                dzdm = dzdp1 * (1 / mean + delta ** 2 / (1 + delta ** 2) / mean) + dzdp2 * (-delta ** 2 / (mean * np.sqrt(np.log(1 + delta ** 2)) * (1 + delta ** 2)))
                dzds = dzdp1 * (-delta / mean / (1 + delta ** 2)) + dzdp2 * (delta / mean / np.sqrt(np.log(1 + delta ** 2)) / (1 + delta ** 2))

            elif distribution_type_i == 3:  # Gamma
                f = st.gamma.cdf(x, par2, 0, 1 / par1)
                z0 = st.norm.isf(1 - f)
                f = st.gamma.cdf(x, par2, 0, 1 / (par1 + delta))
                z1 = st.norm.isf(1 - f)
                f = st.gamma.cdf(x, par2 + delta, 0, 1 / par1)
                z2 = st.norm.isf(1 - f)
                dzdp1 = (z1 - z0) / delta
                dzdp2 = (z2 - z0) / delta
                dzdm = dzdp1 * (1 / standard_deviation ** 2) + dzdp2 * (2 * mean / standard_deviation ** 2)
                dzds = dzdp1 * (-2 * mean / standard_deviation ** 3) + dzdp2 * (-2 * (mean ** 2) / standard_deviation ** 3)

            elif distribution_type_i == 4:  # Shifted Exponential
                f = st.expon.cdf(x, par2, 1 / par1)
                z0 = st.norm.isf(1 - f)
                f = st.expon.cdf(x, par2, 1 / (par1 + delta))
                z1 = st.norm.isf(1 - f)
                f = st.expon.cdf(x, par2 + delta, 1 / par1)
                z2 = st.norm.isf(1 - f)
                dzdp1 = (z1 - z0) / delta
                dzdp2 = (z2 - z0) / delta
                dzdm = dzdp2
                dzds = -dzdp1 / standard_deviation ** 2 - dzdp2

            elif distribution_type_i == 5:  # Shifted Rayleigh
                f = st.rayleigh.cdf(x, par2, par1)
                z0 = st.norm.isf(1 - f)
                f = st.rayleigh.cdf(x, par2, par1 + delta)
                z1 = st.norm.isf(1 - f)
                f = st.rayleigh.cdf(x, par2 + delta, par1)
                z2 = st.norm.isf(1 - f)
                dzdp1 = (z1 - z0) / delta
                dzdp2 = (z2 - z0) / delta
                dzdm = dzdp2
                dzds = dzdp1 / np.sqrt((4 - np.pi) / 2) - dzdp2 * np.sqrt(np.pi / (4 - np.pi))

            elif distribution_type_i == 6:  # Uniform
                f = st.uniform.cdf(x, par1, par2 - par1)
                z0 = st.norm.isf(1 - f)
                f = st.uniform.cdf(x, par1 + delta, par2 - par1 - delta)
                z1 = st.norm.isf(1 - f)
                f = st.uniform.cdf(x, par1, par2 - par1 + delta)
                z2 = st.norm.isf(1 - f)
                dzdp1 = (z1 - z0) / delta
                dzdp2 = (z2 - z0) / delta
                dzdm = dzdp1 + dzdp2
                dzds = -dzdp1 * np.sqrt(3) + dzdp2 * np.sqrt(3)

            Jzt[index, 0] = dzdm
            Jzt[index, 1] = dzds
            Jzt[index, 2] = dzdp1
            Jzt[index, 3] = dzdp2

        for i in range(self.variable_number):
            J = Jzt[i].reshape(1, 4)
            L = np.linalg.inv(self.cholesky_zz)[:, i].reshape(self.variable_number, 1)
            Jut = L @ J
            beta_sensitivity_theta_f[i, :] = alpha @ Jut
        pf_sensitivity_theta_f = -st.norm.pdf(-beta)*beta_sensitivity_theta_f

        return beta_sensitivity_theta_f, pf_sensitivity_theta_f
