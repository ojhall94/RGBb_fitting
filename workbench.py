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

    x = np.arange(-4,5,0.1)
    y = np.arange(-4,5,0.1)
    X, Y = np.meshgrid(x, y)

    sigx = 1.
    sigy = 1.
    mx = 0.
    my = 0.
    rholist = np.arange(-0.9,0.9,0.2)
    for rho in rholist:
        f = 1/(2*np.pi*sigx*sigy*np.sqrt(1-rho**2)) * np.exp(\
            (-1/(2*(1-rho**2))) * (\
            ((X - mx)**2/sigx**2) + ((Y - my)**2/sigy**2)) -\
            (2*rho*(X - mx)*(Y - my))/(sigx * sigy)))


        plt.contour(x, y, f, n=100)
        plt.show()







    sys.exit()
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
