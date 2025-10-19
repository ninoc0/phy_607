####################################################################################
# This file contains methods for sampling from a power-law and Gaussian distribution
from numpy.random import uniform as uniform
import numpy as np

class sampling:

    def gaussian_pdf(x, mu, sigma):
        """
        Gaussian PDF
        """
        prefactor = 1/(np.sqrt(2 * sigma ** 2))
        exponent = ((x-mu)**2)/(2 * sigma ** 2)
        return  prefactor * np.exp(-exponent)

    def sample_gaussian(N, mu, sigma):
        """
        Sample from Gaussian using rejection sampling (favorable, since there is no closed form integral of a Gaussian)
        """
        const = 1/(np.sqrt(2 * sigma ** 2)) #maximum value attained by gaussian PDF
        
        x_prop = uniform(0,1,N) #randomly sample from [0,1] N times. Proposal x dist.
        target_dist = sampling.gaussian_pdf(x_prop, mu, sigma) #evaluate proposal dist

        y_prop = uniform(0,1,N) #random sample from [0,1] to compare to proposal dist
        mask = y_prop <= target_dist/const
        accepted_samples = x_prop[mask]
    
        return accepted_samples

    def sample_power_law(N,k,x_min): 
        """
        Sample from power-law distribution using inverse-CDF method
        """
        uniform_samples = uniform(0,1,N)  #return N uniformly distributed samples between (0,1]
    
        power_law_inv = lambda n,k,x_min: x_min * np.power(n, (-1/(k-1)))    #def inverse CDF

        return power_law_inv(uniform_samples, k, x_min)
