
import numpy as np


class LLModels:
    '''Set of likelihood functions and models'''
    def __init__(self, _x, _y, _labels):
        self.x = _x
        self.y = _y

        self.locb = np.where(np.array(_labels)=='$b$')[0][0]
        self.locsigb = np.where(np.array(_labels)==r"$\sigma(b)$")[0][0]
        self.loclambd = np.where(np.array(_labels)==r"$\lambda$")[0][0]
        self.locsigma = np.where(np.array(_labels)==r"$\sigma$")[0][0]
        self.locm = np.where(np.array(_labels)==r"$m$")[0][0]
        self.locc = np.where(np.array(_labels)==r"$c$")[0][0]

        self.sigx = np.where(np.array(_labels)==r"$\sigma_x$")[0][0]
        self.sigy = np.where(np.array(_labels)==r"$\sigma_y$")[0][0]
        self.my = np.where(np.array(_labels)==r"$\mu_y$")[0][0]
        self.rho = np.where(np.array(_labels)==r"$\rho$")[0][0]


    def gauss_x(self, p):
        '''A simple gaussian in x space'''
        b = p[self.locb]
        sigb = p[self.locsigb]

        #Calculating the likelihood in the X direction
        lnLx = -0.5 * (((b - self.x) / sigb)**2 + 2*np.log(sigb) +np.log(2*np.pi))
        return lnLx

    def exp_x(self, p):
        '''A normalised rising exponential probability in x space'''
        lambd = p[self.loclambd]

        #Calculating the likelihood in the X direction
        A = lambd * (np.exp(lambd*self.x.max()) - np.exp(lambd*self.x.min()))**-1
        lnLx = np.log(A) + lambd*self.x
        return lnLx

    def gauss_line_y(self, p):
        '''A fit in y for a straight line with gaussian noise.'''
        sigma = p[self.locsigma]
        m = p[self.locm]
        c = p[self.locc]

        M = self.x * m + c
        lnLy = -0.5 * (((self.y - M) / sigma)**2 + 2*np.log(sigma) + np.log(2*np.pi))
        return lnLy

    def bivar_gaussian(self, p):
        '''A bivariate gaussian function for probability in two dimensions.'''
        sigx = p[self.locsigx]
        sigy = p[self.locsigy]
        mx = p[self.locb]
        my = p[self.locmy]
        rho = p[self.locrho]

        lnLxy = -np.log(2*np.pi) - np.log(sigx*sigy) - 0.5*np.log(1-rho**2) -\
                (1/(2*(1-rho**2))) * (\
                (X - mx)**2/sigx**2 + (Y - my)**2/sigy**2 -\
                (2*rho*(X - mx)*(Y - my))/(sigx * sigy))

        return lnLx
