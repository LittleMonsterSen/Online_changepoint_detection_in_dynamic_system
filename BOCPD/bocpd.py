#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:32:01 2023

@author: senlin
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as ss

class BaseLikelihood(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for online bayesian changepoint detection.
    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.
    Update theta has **kwargs to pass in the timestep iteration (t) if desired.
    To use the time step add this into your update theta function:
        timestep = kwargs['t']
    """

    @abstractmethod
    def pdf(self, data: np.array):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def update_theta(self, data: np.array, **kwargs):
        raise NotImplementedError(
            "Update theta is not defined. Please define in separate class to override this function."
        )

        
class StudentT(BaseLikelihood):
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution
        Parameters:
            alpha - alpha in gamma distribution prior
            beta - beta inn gamma distribution prior
            mu - mean from normal distribution
            kappa - variance from normal distribution
        """

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        return ss.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


def constant_hazard(lam, r):
    """
    Hazard function for bayesian online learning
    Arguments:
        lam - inital prob
        r - R matrix
    """
    return 1 / lam * np.ones(r.shape)


def bocpd(data,model,lam):
    #### data: input data, column represent feature. data rows better not excceed 10^4, it's a simple verion of R
    #### model: probability distribution, we use student's t distribution with parameter mu,kappa,beta, alpha
    #### lam: lambda for constant hazard function, means change points arrival rate is 1/lambda
    import numpy as np
    from collections import defaultdict

    maxes = np.zeros(len(data) + 1)
    dims = data.shape[1] # dimension o
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    cps = np.zeros(1)
    most_likely_sets = np.zeros(2)
    maxCP = defaultdict()
    maxCP[0]=[]
    maxCP[1]=[0]

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.

        predProbsMat = np.empty((dims,t+1))
        for dimi in range(dims):
            # calculate predictive probability for each dimension
            predProbsMat[dimi] = np.log(model[dimi].pdf(x[dimi]))

            # Update sufficient statistics.
            model[dimi].update_theta(x[dimi])


        # Evaluate the hazard function for this interval
        H = constant_hazard(lam,np.array(range(t + 1)))
        
        # pretend each dimension are uncorrelated. We use beta-TCVAE to achieve this.
        predprobs = np.exp(predProbsMat.sum(0))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])


        maxes[t] = R[:, t].argmax()
        
        # column maximum as cps
        cps = np.unique(np.concatenate((cps,[t-R[:-2, t].argmax()])))
        
        # cp with highest probability
        currprobs = most_likely_sets + np.log(R[0:t+2, t + 1])[::-1]
        most_likely_sets = np.concatenate((most_likely_sets,[np.max(currprobs)]))
        maxCP[t+2] = np.concatenate((maxCP[currprobs.argmax()],[currprobs.argmax()]))
        
    return R, maxCP[t+2]