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
import cLLModels
import cMCMC

def get_values(style):
    if style == 'rgb':
        files = glob.glob('data/RGB_data.txt')[0]
        df = pd.read_table(files, sep=',', header=0, error_bad_lines=False)

        df['logT'] = np.log10(df.Teff)
        df['logL'] = np.log10(df.L)
        df['lognumax'] = np.log10(df.numax)

        df = df.sort_values(by=['numax'])
        # df = df[df.lognumax <= 2.0]

    if style == 'trilegal':
        files = glob.glob('data/trilegal_keplerfov.txt')[0]
        df = pd.read_table(files, sep=',', header=0, error_bad_lines=False)

        df['logT'] = np.log10(df.teff)
        df['lognumax'] = np.log10(df.numax)

        df = df.sort_values(by=['numax'])
        df = df[(df.lognumax > 1.) & (df.lognumax < 2.)]

    return df.lognumax, df.logT, df

class cLikelihood:
    '''A likelihood function that pulls in log likehoods from the LLModels class
    '''
    def __init__(self,_lnprior, _Model):
        self.lnprior = _lnprior
        self.Model = _Model

    #Likelihood for the 'foreground'
    def lnlike_fg(self, p):
        return self.Model.bivar_gaussian(p)

    def lnlike_bg(self, p):
        return self.Model.tophat_x(p) + self.Model.gauss_line_y(p)

    def lnprob(self, p):
        Q = p[-1]

        # First check the prior.
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = self.lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = self.lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.nansum(np.logaddexp(arg1, arg2))

        return lp + ll

    def __call__(self, p):
        logL = self.lnprob(p)
        return logL

def probability_plot(x, y, fy, X, Y, bins, exp_x, line_y, bi_g, bi_x, bi_y):
    #Plotting residuals with histograms
    left, bottom, width, height = 0.1, 0.35, 0.60, 0.60
    fig = plt.figure(1, figsize=(8,8))
    sax = fig.add_axes([left, bottom, width, height])
    yax = fig.add_axes([left+width+0.02, bottom, 0.25, height])
    xax = fig.add_axes([left, 0.1, width, 0.22], sharex=sax)
    lax = fig.add_axes([left+width+0.02, 0.1, 0.25, 0.22])
    sax.xaxis.set_visible(False)
    yax.set_yticklabels([])
    yax.set_xticklabels([])
    lax.set_yticklabels([])
    xax.set_yticklabels([])
    xax2 = xax.twinx()
    xax2.set_yticklabels([])
    xax.grid()
    xax.set_axisbelow(True)
    yax.grid()
    yax.set_axisbelow(True)
    lax.grid()
    lax.set_axisbelow(True)

    fig.suptitle('Probability functions to be applied to APOGEE data.')

    sax.hist2d(x, y,bins=bins, cmap='Blues_r', zorder=1000)
    sax.plot(x,fy,c='r',linestyle='--',label='Straight line fit', zorder=1001)
    c4 = sax.contour(X,Y,bi_g, cmap='copper',alpha=.5,label='Bivariate Gaussian',zorder=1001)

    yax.hist(y,bins=bins,color='r',histtype='step',orientation='horizontal', normed=True)
    yax.scatter(bi_y,y,s=5,c='cornflowerblue',alpha=.5,label='Bivariate in Y')
    yax.set_ylim(sax.get_ylim())
    yax.legend(loc='best')


    xax2.scatter(x,bi_x,s=5,c ='cornflowerblue', alpha=.5,label='Bivariate in X')
    xax.scatter(x,exp_x,s=5,c='orange', alpha=.5,label='RGB Model in X')
    xax.hist(x,bins=bins,histtype='step',color='r',normed=True)
    h1, l1 = xax.get_legend_handles_labels()
    h2, l2 = xax2.get_legend_handles_labels()
    xax.legend(h1+h2, l1+l2)

    lax.hist(y-fy,bins=bins,histtype='step',color='r',normed=True)
    lax.scatter(y-fy, line_y, s=5,c='orange',alpha=.5, label='RGB Model in Y')
    lax.set_xlabel(r"log$_{10}$($T_{\rm{eff}}$) - Line Fit")
    lax.axvline(0.0,c='r',linestyle='--',label='Line Fit')
    lax.legend(loc='best')

    sax.set_ylabel(r"log$_{10}$($T_{\rm{eff}}$)")
    xax.set_xlabel(r"log$_{10}$($\nu_{\rm{max}}$)")

    return fig

if __name__ == '__main__':
    plt.close('all')
    mlo = np.arange(0.9,1.7,0.2)
    mhi = np.arange(1.1,1.9,0.2)
    flo = np.arange(-0.5,0.3,0.2)
    fhi = np.arange(-0.3,0.5,0.2)

    x, y, df = get_values('trilegal')

####---SETTING UP DATA

    bins = int(np.sqrt(len(x)))

    #Plotting the data to be fit to
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].scatter(x, y, s=3, zorder=1000)
    ax[0].set_title(r'APOGEE data')
    ax[0].set_ylabel(r"$log_{10}$($T_{eff}$ (K))")
    ax[0].legend(loc='best',fancybox=True)

    #Making first guess for mu_x
    n, b = np.histogram(10**x,bins=bins)
    lnuguess = np.log10(b[np.argmax(n)])

    ax[1].hist(x, bins=bins, color ='k', histtype='step', normed=1)
    ax[1].axvline(lnuguess,c='r',label=r"$\nu_{max}$ estimate")
    ax[1].set_title(r"Histogram in $log_{10}$($\nu_{max}$)")
    ax[1].set_xlabel(r"$log_{10}$($\nu_{max}$ ($\mu$Hz))")
    fig.tight_layout()
    fig.savefig('Output/investigate_RGB.png')
    plt.close('all')

####---BUILDING KDE AND OTHER PARAM ESTIMATES

    #Making first guess for x, y
    fn = np.polyfit(x, y, 1)
    fy = x*fn[0]+fn[1]

    #Getting the KDE of the 2D distribution
    xxyy = np.ones([len(x),2])
    xxyy[:,0] = x
    xxyy[:,1] = y
    kde = stats.gaussian_kde(xxyy.T)

    #Setting up a 2D meshgrid
    size = 200
    xx = np.linspace(x.min(),x.max(),size)
    yy = np.linspace(y.min(),y.max(),size)
    X, Y  = np.meshgrid(xx, yy)
    d = np.ones([size, size])

    #Calculating the KDE value for each point on the grid
    for idx, i in tqdm(enumerate(xx)):
        for jdx, j in enumerate(yy):
            d[jdx, idx] = kde([i,j])

    #Making first guess for mu_y
    lteffguess = Y.ravel()[np.argmax(d)]

    '''HARD CODING IN SOME GUESS HERE FOR THE TRILEGAL SIM'''
    lnuguess = 1.5


####---SETTING UP MCMC
    labels_mc = [r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$",\
                "m","c",r"$\sigma$",\
                "$Q$"]
    start_params = np.array([lnuguess, lteffguess, 0.07, np.std(y), -0.8,\
                            fn[0], fn[1], np.std(y-fy),\
                            0.5])
    bounds = [(lnuguess-.1, lnuguess+.1,), (lteffguess-0.05,lteffguess+0.05),\
                (0.01,0.1), (0.1*np.std(y), 1.5*np.std(y)), (-1., 0.),\
                (fn[0]*0.8, fn[0]*1.2), (fn[1]*0.8, fn[1]*1.2),\
                (np.std(y-fy)*0.5, np.std(y-fy)*1.5),\
                (0,1)]
####---CHECKING MODELS BEFORE RUN
    #Getting meshgrid version of bivariate model
    ModeLLs = cLLModels.LLModels(X, Y, labels_mc)
    bi_g = np.exp(ModeLLs.bivar_gaussian(start_params))

    #Getting other probability functions
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    # exp_x = np.exp(ModeLLs.exp_x(start_params))
    tophat_x = np.exp(ModeLLs.tophat_x(start_params))
    line_y = np.exp(ModeLLs.gauss_line_y(start_params))
    bi_x, bi_y = ModeLLs.return_bivar_sologauss(start_params)

    fig = probability_plot(x, y, fy, X, Y, bins, tophat_x, line_y, bi_g, bi_x, bi_y)
    fig.savefig('Output/visual_RGB.png')
    plt.show()
    plt.close('all')

####---RUNNING MCMC
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    lnprior = cPrior.Prior(bounds)
    Like = cLikelihood(lnprior,ModeLLs)

    ntemps, nwalkers = 4, 32

    Fit = cMCMC.MCMC(start_params, Like, lnprior, 'none', ntemps, 1000, nwalkers)
    chain = Fit.run()

####---CONSOLIDATING RESULTS
    corner.corner(chain, bins=35,labels=labels_mc)
    plt.savefig('Output/corner_RGB.png')
    plt.close()

    lnK, fg_pp = Fit.log_bayes()
    mask = lnK > 1
    Fit.dump()

####---PLOTTING RESULTS
    print('Plotting results...')
    npa = chain.shape[1]
    res = np.zeros(npa)
    std = np.zeros(npa)
    for idx in np.arange(npa):
        res[idx] = np.median(chain[:,idx])
        std[idx] = np.std(chain[:,idx])

    resy = res[ModeLLs.locm]*x + res[ModeLLs.locc]

    #Getting meshgrid version of bivariate model
    ModeLLs = cLLModels.LLModels(X, Y, labels_mc)
    bi_g = np.exp(ModeLLs.bivar_gaussian(res))

    #Getting other probability functions
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    # exp_x = np.exp(ModeLLs.exp_x(start_params))
    tophat_x = np.exp(ModeLLs.tophat_x(start_params))
    line_y = np.exp(ModeLLs.gauss_line_y(res))
    bi_x, bi_y = ModeLLs.return_bivar_sologauss(res)

    fig = probability_plot(x, y, resy, X, Y, bins, tophat_x, line_y, bi_g, bi_x, bi_y)
    fig.savefig('Output/visual_result_RGB.png')
    plt.show()
    plt.close('all')

    #Plotting results
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_axisbelow(True)

    col = ax.scatter(x, y, c=fg_pp, s=5, cmap='viridis')
    ax.plot(x, resy, c='r',linestyle='--',label='Line Fit')
    fig.colorbar(col, ax=ax, label = 'RGBb membership posterior probability')

    sax.set_ylabel(r"log$_{10}(T_{\rm{eff}})$")
    xax.set_xlabel(r"log$_{10}(\nu_{\rm{max}})$")
    fig.savefig('Output/result_RGB.png')
    plt.show()
    plt.close('all')


    #Plotting identified RGBB stars
    fig, ax = plt.subplots()
    ax.scatter(df.Teff[mask], df.numax[mask], c='y', s=3, label='RGBB Stars')
    ax.scatter(df.Teff[~mask], df.numax[~mask], c='g', s=3, label='RGB Stars')
    ax.legend(loc='best',fancybox=True)
    ax.text(4600,150,r"$\nu_{max}$ RGBB stddev = "+str.format('{0:.3f}',np.std(df.numax[mask]))+r"$\mu$Hz")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(r"Identified RGBB stars in $\nu_{max}$ for real data.")
    ax.set_xlabel(r"$T_{eff}$ (K)")
    ax.set_ylabel(r"$\nu_{max}$($\mu$Hz)")
    fig.savefig('Output/comparison_RGB.png')
    plt.close('all')


    #Saving out the data with new labels
    df['label'] = ''
    df.label[mask] = 'RGBB'
    df.label[~mask] = 'RGB'

    header = df.columns
    df.to_csv('RGB_labeled.txt',header=header,sep='\t')
