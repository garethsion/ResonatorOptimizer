#!/usr/bin/env

"""
    ConformalMapping contains the methods for calculating the Schwartz-Christofell 
    mapping functions for a CPW geometry. 
"""

import numpy as np
import pandas as pd
import scipy.constants as spc 
from scipy.special import ellipk

class ConformalMapping:
    """
        ConformalMapping contains the methods for calculating the Schwartz-Christofell 
        mapping functions for a CPW geometry. 
    """
    def __init__(self,width=0,gap=0,er=0,h=None,t=0):
        """
        Constructor method - Initializes the cpw geometry

        :type width: float
        :param width: conductor width

        :type gap: float
        :param gap: gap between conductor and ground plane

        :type er: float
        :param er: relative permittivity of substrate

        :type h: float
        :param h: thickness of substrate

        :type t: float
        :param t: thickness of conductor thin film
        """

        self.__w = width
        self.__s = gap
        self.__er = er
        self.__h = h
        self.__t = t
        
        if not self.__h:
            self.__eeff = (er + 1) /2
        elif self.__h:
            self.__eeff = self.effective_permittivity()


    def elliptic_integral(self,h=None):
        """
        calculates the complete elliptic integral of the first kind
        for a given cpw geometry as part of a conformal mapping strategy.

        :type h: float
        :params h: substrate thickness (opt)
        """
        if not self.__h:
            k = self.__w / (self.__w + 2*self.__s)
            kp = np.sqrt(1-k**2)
        elif self.__h:            
            k = ( np.sinh((np.pi*self.__w)/(4*self.__h)) 
                 / np.sinh( (np.pi*(self.__w+2*self.__s)) 
                           / (4*self.__h) ) )
            kp = np.sqrt(1-k**2)
        Kk = ellipk(k)
        Kkp = ellipk(kp)
        return (Kk,Kkp)

    def effective_permittivity(self):
        """
        calculates the effective permittivity by performing complete 
        elliptic integral of the first kind 
        """
        Kk1,Kkp1 = self.elliptic_integral()
        Kk2,Kkp2 = self.elliptic_integral(h=self.__h)
        
        eeff = 1 + .5*(self.__er-1) * Kk2/Kkp2 * Kkp1/Kk1
        return eeff
        
    def g(self):
        """
        calculates the geometric factor necessary for calculating the
        kinetic inductance of a CPW
        """
        w = self.__w
        s = self.__s
        t = self.__t
        
        k = (w) / (w+(2*s))
        Kk,Kkp = self.elliptic_integral()
        
        outer = 1 / (2*(k**2)*(Kk**2))
        inner1 = -np.log(t / (4*w)) 
        inner2 = - (w/(w+(2*s))) * np.log(t / (4*(w+2*s)) )
        inner3 = (2*(w+s)/(w+(2*s))) * np.log(s / (w+s))
        inner = inner1 + inner2 + inner3
        g = outer * inner
        return g