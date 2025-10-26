####################################################################################
# This file contains methods for sampling from a power-law and Gaussian distribution
from numpy.random import uniform as uniform
import numpy as np


class sampling:

    def gaussian_pdf(x, mu, sigma):
        """
        Gaussian PDF

        Arguments:
        x - discrete points at which to sample Gaussian PDF
        mu - mean of Gaussian PDF
        sigma - standard deviation of Gaussian PDF

        Returns:
        points distributed according to Gaussian PDF
        """

        prefactor = 1 / (np.sqrt(2 * sigma**2))
        exponent = ((x - mu) ** 2) / (2 * sigma**2)
        return prefactor * np.exp(-exponent)

    def sample_gaussian(N, mu, sigma):
        """
        Sample from Gaussian using rejection sampling (favorable, since there is no closed form integral of a Gaussian)

        Arguments:
        N - Number of samples desired
        mu - mean of target Gaussian
        sigma - standard deviation of target Gaussian

        Returns:
        Array of accepted samples, distributed according to target Gaussian PDF
        """

        const = 1 / (np.sqrt(2 * sigma**2))  # maximum value attained by gaussian PDF

        accepted_samples = []  # list to store accepted samples

        while (
            len(accepted_samples) < N
        ):  # while length of accepted samples list is lt desired number of samples
            x_prop = uniform(
                0, 1
            )  # randomly sample from [0,1] N times. Proposal x dist.
            target_dist = sampling.gaussian_pdf(
                x_prop, mu, sigma
            )  # evaluate proposal dist

            y_prop = uniform(
                0, 1
            )  # random sample from [0,1] to compare to proposal dist
            if y_prop <= target_dist / const:
                accepted_samples.append(x_prop)

        return np.array(accepted_samples)

    def power_law_pdf(x, a):
        """
        Power law PDF

        Arguments:
        x - discrete points at which to sample PDF
        a - exponent of PDF, necessary that a>0

        Returns:
        points distributed according to power law PDF
        """

        return a * x ** (a - 1)

    def sample_power_law(N, a):
        """
        Sample from power-law distribution using inverse-CDF method

        Arguments:
        N - Number of samples desired
        a - exponent of power law, necessary that a>0

        Returns:
        Samples distributed according to power law
        """

        uniform_samples = uniform(
            0, 1, N
        )  # return N uniformly distributed samples between (0,1]

        power_law_inv = lambda n, a: n ** (1 / a)  # def inverse CDF

        return power_law_inv(uniform_samples, a)
