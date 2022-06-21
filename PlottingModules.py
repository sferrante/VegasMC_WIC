import numpy as np
import matplotlib.pyplot as plt
from functools import partial 
import WIC_Pheno_Formulae as WIC
from WIC_Pheno_Formulae import *


def ScaleFactors(rescaled_hist, nbins, integral):
    new_width = max(rescaled_hist) - min(rescaled_hist)
    return (nbins/new_width)*np.full(len(rescaled_hist), integral/len(rescaled_hist))


#######################################################
##################   e+e- --> μ1μ1'  ##################
##################     Plotting      ##################
##################       dσdμ        ##################
#######################################################
def Plot_σ_ee(dictionaries, μ0, sqrtS, μϼ, nbins, scale, SCIPY, LEGEND, TITLE, SIZE):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = SIZE);
    if TITLE==True:
        title = "Unweighted Histograms of %i (Initial) Integration Points in " %50000
        fig.suptitle(title + r" $\frac{d^{2}σ}{dμ_{1}dμ_{1}^{'}}$", size=20, y=1.0)
    colors=['blue']
    ################################# Vegas MC #################################
    for i, d in enumerate(dictionaries):
        dσdμ1_events  = d['μ1 Events']
        dσdμ1p_events = d['μ1p Events']
        integral_σ    = d['Integral']
        ScaleFs_μ1    = ScaleFactors(dσdμ1_events, nbins, integral_σ)
        ScaleFs_μ1p   = ScaleFactors(dσdμ1p_events, nbins, integral_σ)
        ############ PLOT ############
        ax1.hist(dσdμ1_events,  weights=ScaleFs_μ1,  bins=nbins, density=False, histtype='step', color=colors[i])
        ax2.hist(dσdμ1p_events, weights=ScaleFs_μ1p, bins=nbins, density=False, histtype='step', color=colors[i])
    ################################# SCIPY check ... #################################
        if SCIPY==True:
            μ1s = np.linspace(100, 400, 100)
            dσdμ = partial(WIC.dσdμ, μ0=μ0, sqrtS=sqrtS, μϼ=μϼ)
            dσdμp = partial(WIC.dσdμ, μ0=μ0, sqrtS=sqrtS, μϼ=μϼ)
            ax1.plot(μ1s, np.array(list(map(dσdμ, μ1s))), color='k')
            ax2.plot(μ1s, np.array(list(map(dσdμp, μ1s))), color='k')
    ###################################################################################
    ax1.set_xlabel(r'$μ_{1}$', size=15)
    ax2.set_xlabel(r"$μ_{1}^{'}$", size=15)
    if LEGEND==True:
        ax1.legend([r'$\,\,\frac{dσ}{dμ_{1}}$ (ab/GeV)'], fontsize=14);
        ax2.legend([r"$\,\,\frac{dσ}{dμ_{1}^{'}}$ (ab/GeV)"], fontsize=14);
#         ax1.legend(['Unweighted (%i)'%len(dσdμ1_events),  r'$\,\,\frac{dσ}{dμ_{1}}$ (ab/GeV)'], fontsize=14);
#         ax2.legend(['Unweighted (%i)'%len(dσdμ1p_events), r"$\,\,\frac{dσ}{dμ_{1}^{'}}$ (ab/GeV)"], fontsize=14);
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    
    
   
    
#######################################################
##################   μ1 --> Z + μ2   ##################
##################     Plotting      ##################
##################      dΓdμ2        ##################
#######################################################
def Plot_2body(dictionaries, μ1s, μ0, sqrtS, μϼ, nbins, scale):
    fig, ax1 = plt.subplots(1, 1, figsize = (8,6));
    title = "Unweighted Histograms of %i (Initial) Integration Points in" %100000
    fig.suptitle(title + r" $\frac{d\,Γ_{2 body}}{dμ_{2}}$", size=20, y=1.0)
    colors = ['orange','green','blue','red','purple']
    for i, d in enumerate(dictionaries):
        dΓ2dμ2_events = d['μ2 Events']
        integral_Γ2   = d['Integral']
        ScaleFScaleFs_μ2 = ScaleFactors(dΓ2dμ2_events, nbins, integral_Γ2)
        ############ PLOT ############
        ax1.hist(dΓ2dμ2_events, weights = ScaleFScaleFs_μ2, bins=nbins, 
                 density=False, histtype='step',color=colors[i], label='%i'%μ1s[i])
        ################################# Numpy check ... #################################
        μ1_i = μ1s[i]
        dΓ2_dμ2 = partial(WIC.dΓ2_dμ2, μ1=μ1_i, μ0=μ0, μϼ=μϼ)
        μ2s = np.linspace(100, μ1_i-mZ, 100)
        ax1.plot(μ2s, np.array(list(map(dΓ2_dμ2, μ2s))), color='k',alpha=0.5)
        ###################################################################################
    ax1.set_xlabel(r'$μ_{2}$', size=15)
    ax1.set_yscale(scale)
    ax1.legend()
    
    
##############################################################
##################   μ1 --> fbar + f + μ2   ##################
##################        Plotting          ##################
##################     d3Γ_dμ2dx3dx4        ##################
##############################################################
def Plot_3body(dictionaries, figsize, nbins, TITLE, COLORS, LEGEND, ylabel, xlabel1, xlabel2, xlabel3, labelsize, ticksize, legendsize, scale):
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = figsize)
    if TITLE == True:
        title = "Unweighted Histograms of %i (Initial) Integration Points in " %200000
        fig.suptitle(title + r"$\frac{d^{3}Γ}{dμ_{2}dx_{3}dx_{4}}$", size=20, y=1.0)
#     mpl.style.use('default')
    for i in range(len(dictionaries)):
        dΓ3dμ2_events = dictionaries[i]['μ2 Events']
        dΓ3dx3_events = dictionaries[i]['x3 Events']
        dΓ3dx4_events = dictionaries[i]['x4 Events']
        integral_Γ3   = dictionaries[i]['Integral']
        ScaleFs_μ2 = ScaleFactors(dΓ3dμ2_events, nbins, integral_Γ3)
        ScaleFs_x3 = ScaleFactors(dΓ3dx3_events, nbins, integral_Γ3)
        ScaleFs_x4 = ScaleFactors(dΓ3dx4_events, nbins, integral_Γ3)
        ############ PLOT ############
        ax1.hist(dΓ3dμ2_events, weights = ScaleFs_μ2, bins=nbins, density=False, histtype='step', color=COLORS[i])
        ax2.hist(dΓ3dx3_events, weights = ScaleFs_x3, bins=nbins, density=False, histtype='step', color=COLORS[i])
        ax3.hist(dΓ3dx4_events, weights = ScaleFs_x4, bins=nbins, density=False, histtype='step', color=COLORS[i])
    ax1.legend(LEGEND, fontsize=legendsize); ax2.legend(LEGEND, fontsize=legendsize); ax3.legend(LEGEND, fontsize=legendsize); 
    ax1.set_ylabel(ylabel, size=labelsize)
    ax1.set_xlabel(xlabel1, size=labelsize)
    ax2.set_xlabel(xlabel2, size=labelsize)
    ax3.set_xlabel(xlabel3, size=labelsize)
    ax1.tick_params(labelsize=ticksize); ax2.tick_params(labelsize=ticksize); ax3.tick_params(labelsize=ticksize)
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    ax3.set_yscale(scale)
    
    
    
    
    
    
    
    