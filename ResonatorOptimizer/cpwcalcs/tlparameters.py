import numpy as np
import pandas as pd
from ResonatorOptimizer.cpwcalcs import conformalmapping as cm

class TLParameters:

    def __init__(self,width=0,gap=0,er=0,h=None,t=0):
        
        self.__w = width
        self.__s = gap
        self.__er = er
        self.__h = h
        self.__t = t

        self.__cm = cm.ConformalMapping(width=self.__w,
                gap=self.__s,er=self.__er,h=self.__h,t=self.__t)

    def alpha(self,wavelength=0,tan_d=0.005):
        eeff = self.__cm.effective_permittivity()
        ad = (self.__er/np.sqrt(eeff)) * ((eeff-1)/(self.__er-1)) * (np.pi/wavelength) * tan_d
        return ad

    def beta(self,freq):
        Ll = self.total_inductance_per_length()
        Cl = self.capacitance_per_length()
        return 2*np.pi*freq*np.sqrt(Ll*Cl)

    def gamma(self,freq,tan_d=0.005):
        alpha = self.alpha(tan_d)
        beta = self.beta(freq)
        return alpha + 1j*beta