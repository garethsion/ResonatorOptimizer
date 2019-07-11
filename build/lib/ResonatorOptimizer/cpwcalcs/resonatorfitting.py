import lmfit
from lmfit.models import BreitWignerModel,LinearModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class ResonatorFitting:
    def __init__(self):
        return
    
    def LorentzianFit(self,freq,trace, plot = True):
        
        if np.any(np.iscomplex(trace)):
            trace = trace.real
        
        #print (len(trace))
        start,stop = None, None                                         #Specifies the window within the data to analyse.
        Lin_mod = LinearModel()                                         #Linear lmfit model for background offset and slope
        BW_mod = BreitWignerModel()                                     #Breit-Wigner-Fano model
        mod = BW_mod+Lin_mod
        
        x = freq[start:stop]/1E6                                        #Convert frequencies to MHz
        trace = (10**(trace/10))                                        #Convert decibel data to linear
        y = trace[start:stop]
        
        pars = BW_mod.guess(y, x=x)                                     #Initialize fit params
        pars += Lin_mod.guess(y,x=x, slope = 0, vary = False)           
        pars['center'].set(value=x[np.argmax(y)], vary=True, expr='')   #Find the highest transmission value. Corresponding frequency is used as a guess for the centre frequency
        pars['sigma'].set(value=0.05, vary=True, expr='')               #Linewidth
        pars['q'].set(value=0, vary=True, expr='')                      #Fano factor (asymmetry term). q=0 gives a Lorentzian
        pars['amplitude'].set(value=-0.03, vary=True, expr='')          #Amplitude

        out  = mod.fit(y,pars,x=x)
#         print (out.fit_report())
        #print (out.params['amplitude'],out.params['q'],out.params['sigma'])
        sigma = out.params['sigma']
        centre = out.params['center']
        
        dic = {'x':x,'y':y,'fit':out.best_fit,'out':out,'sigma':sigma.value,
               'centre':centre.value,'Q':centre.value/sigma.value}
        
        df = pd.DataFrame(data=dic)

        if plot == True:
            print(out.params['amplitude'],out.params['q'],out.params['sigma'])
            plt.plot(x,y, color = 'orange', label = 'Data')
            plt.plot(x, out.best_fit, color = 'darkslateblue',label = 'Fano resonance fit')

#         return(sigma.value,centre.value,centre.value/sigma.value)       #Returns linewidth in GHz, centre in GHz and Q factor
        return df
