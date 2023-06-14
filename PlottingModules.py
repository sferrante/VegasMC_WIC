import numpy as np
import matplotlib.pyplot as plt
from functools import partial 
import WIC_Pheno_Formulae as WIC
from WIC_Pheno_Formulae import *


## Defining the histogram weights 'ScaleFactors' according to the min/max of the data ##
def ScaleFactors(rescaled_hist, nbins, integral):
    new_width = max(rescaled_hist) - min(rescaled_hist)
    return (nbins/new_width)*np.full(len(rescaled_hist), integral/len(rescaled_hist))
class Plotter:
    def __init__(self, name, NProjections, SIZE, nbins):
        self.name = name; self.NProjections = NProjections;
        self.SIZE = SIZE; self.nbins = nbins
        self.dictionaries = []
        self.fig, self.ax = plt.subplots(1, np.sum(NProjections), figsize = SIZE);
    def addPlot(self, dictionaryList, colors, scale):
        NProjections = self.NProjections;
        SIZE = self.SIZE; nbins = self.nbins
        self.dictionaries.append(dictionaryList)
        ## looping through dictionaries 'd' (..if factorizable. usually, d=1) ##
        PlotCount=-1;
        for d in range(len(dictionaryList)):
        ## looping through projections 'i' ##
            ProjectionCount=0;  
            for i in range(NProjections[d]):
                events = np.transpose(dictionaryList[d]['x Events'])[ProjectionCount]
                PlotCount+=1
                ProjectionCount+=1
                integral = dictionaryList[d]['Integral']
                ## if there's 1 projection, there's just one 'ax', else 'ax[0,1,..]'
                if np.sum(NProjections)==1: plot=self.ax
                if np.sum(NProjections)!=1: plot=self.ax[PlotCount]
                plot.hist(events, 
                          weights = ScaleFactors(events, nbins, integral), 
                          bins = nbins, density=False, histtype='step', color=colors[d][i] )
                plot.set_yscale(scale)
        plt.tight_layout()
    def addFeatures(self, dictionaryList, xlabels, legend, legendlabel):
        NProjections = self.NProjections; 
        PlotCount=-1;
        for d in range(len(dictionaryList)):
            for i in range(NProjections[d]):
                PlotCount+=1;
                if np.sum(NProjections)==1: plot=self.ax
                if np.sum(NProjections)!=1: plot=self.ax[PlotCount]
                if legend==True: plot.legend(legendlabel[PlotCount], fontsize=14)
                plot.set_xlabel(xlabels[PlotCount], size=15)