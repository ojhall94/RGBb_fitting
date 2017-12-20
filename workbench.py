 # !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd

import glob
import sys

import corner as corner
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

import scipy.stats as stats

import cPrior
import cLikelihood
import cMCMC
import cLLModels

def get_values(US):
    if US == 'RGB':
        files = glob.glob('data/'+US+'*.*')[0]
        df = pd.read_table(files, sep=',', header=0, error_bad_lines=False)

    else:
        files = glob.glob('data/*'+US+'.*')[0]
        df = pd.read_table(files, sep='\s+', header=0, skiprows=3, error_bad_lines=False)

    df['logT'] = np.log10(df.Teff)
    df['logL'] = np.log10(df.L)
    df['lognumax'] = np.log10(df.numax)



    df = df.sort_values(by=['numax'])
    return df.lognumax, df.logT, df



if __name__ == "__main__":
    plt.close('all')
    x, y, df = get_values('RGB')
    size = 1000
    xx = np.linspace(x.min(),x.max(),size)
    yy = np.linspace(y.min(),y.max(),size)
    X, Y  = np.meshgrid(xx, yy)

    labels_mc = [r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$",\
                r"$\lambda$","m","c",r"$\sigma$",\
                "$Q$"]

    start_params = np.array([1.65, 3.67, 0.07, 0.015, -.8,\
                            1.8, 0.04, 3.6, 0.01,\
                            0.5])

    ModeLLs = cLLModels.LLModels(X, Y, labels_mc)
    fg = np.exp(ModeLLs.bivar_gaussian(start_params))

    plt.scatter(x, y, s=5, alpha=.5)
    plt.contour(X, Y, fg)
    plt.show()

    sys.exit()

    xx = np.arange(-5,5,0.2)
    yy = np.arange(-5,5,0.2)
    x, y = np.meshgrid(xx, yy)

    mx = 0.
    my = 0.
    sigy = 3.
    sigx = 0.1
    rho = -.8

    lnLxy = -np.log(2*np.pi) - np.log(sigx*sigy) - 0.5*np.log(1-rho**2) -\
            (1/(2*(1-rho**2))) * (\
            (x - mx)**2/sigx**2 + (y - my)**2/sigy**2 -\
            (2*rho*(x - mx)*(y - my))/(sigx * sigy))

    f = np.exp(lnLxy)
    plt.contour(x, y, f)
    plt.show()

    sys.exit()
    x, y, df = get_values('RGB')

    mlo = np.arange(0.9,1.7,0.2)
    mhi = np.arange(1.1,1.9,0.2)
    flo = np.arange(-0.5,0.3,0.2)
    fhi = np.arange(-0.3,0.5,0.2)

    for lo1, hi1 in zip(mlo,mhi):
        for lo2, hi2 in zip(flo,fhi):
            sel = np.where((df.mass > lo1) & (df.mass < hi1) & (df.FeH > lo2) & (df.FeH < hi2))[0]

            plt.scatter(x[sel],y[sel])
            plt.title('Mass: ('+str(lo1)+'-'+str(hi1)+') | FeH: ('+str(lo2)+'-'+str(hi2)+')')
            plt.show()
