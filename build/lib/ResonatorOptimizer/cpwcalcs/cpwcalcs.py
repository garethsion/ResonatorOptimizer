import numpy as np
import pandas as pd
import scipy.constants as spc 
from scipy.special import ellipk

class cpwCalcs:
    """ cpwCalcs contains the methods necessary for calculating certain parameters of 
    interest of a superconducting cpw structure. Solutions for the resonant frequency, 
    characteristic impedance, phase constant, etc, are determined by solving the
    cpw geometry analytically through conformal mapping.

    CHANGE_LOG
    ----------

    * 29/07/2019 - Changed resonant frequency method to accept different wavelength 
                   cavities
                 - Changed constructor method to accep a specification of the cpw
                   desired electrical length.

    TO_DO
    -----

    * 29/07/2019 - Ensure that the resonant frequency is only calculated for the correct 
                   wavelength cpw
                 - Include a method for printing out a snapshot of the cpw params
                 - Split class into two seperate classes, one for conformal mapping, 
                   the other for calcs
    """
    def __init__(self,width=0,gap=0,length=0,elen=180,fo=0,er=0,h=None,t=0,pen_depth=None):
        """ Constructor method. 

        params: width       : conductor width
                gap         : gap between conductor and ground plane
                length      : conductor length
                elen        : conductor electrical length (degrees)
                fo          : designed resonant frequency
                er          : relative permittivity of substrate
                h           : thiockness of substrate
                t           : thickness of conductor thin film
                pen_depth   : magnetic penetration depth
        """
        self.__w = width
        self.__s = gap
        self.__l = length
        self.__elen = elen
        self.__fo = fo
        self.__er = er
        self.__h = h
        self.__t = t
        self.__pen_depth=pen_depth
        
        if not self.__h:
            self.__eeff = (er + 1) /2
        elif self.__h:
            self.__eeff = self.effective_permittivity()

        # print('CPW with electrical length = ' + str(elen) + ' degrees')

    ######## PRINTING
    
    def print_cpw_params(self):
        """ cpw_params returns the geometric parameters of the cpw structure.

        returns     : pandas dataframe
        """
        dic = {'width':self.__w, 'gap':self.__s, 'length':self.__l,
        'h':self.__h, 't':self.__t, 'er': self.__er, 'eeff':self.__eeff,
        'pen_depth':self.__pen_depth}

        df = pd.DataFrame(data=[dic])

        return df

    def print_wave_params(self):
        dic = {
        'fo':self.resonant_freq(),
        'wavelength':self.wavelength(),
        'vp':self.phase_velocity(),
        'phase_const':self.phase_constant()
        }

        return pd.DataFrame(data=[dic])

    def print_electrical_params(self):
        dic = {
        'Lk':self.Lk(),
        'Ltotal':self.total_inductance_per_length(),
        'Ll':self.geometric_inductance_per_length(),
        'Cl':self.capacitance_per_length(),
        'Z':self.impedance_geometric(),
        'Zki':self.impedance_kinetic(),
        }

        return pd.DataFrame(data=[dic])

    ######## CONFORMAL MAPPING

    def elliptic_integral(self,h=None):
        """elliptic_integral calculates the complete elliptic integral of the first kind
        for a given cpw geometry as part of a conformal mapping strategy.

        params  : h     : substrate thickness (opt)

        returns : (Kk, Kkp)     : tuple 
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
        Kk1,Kkp1 = self.elliptic_integral()
        Kk2,Kkp2 = self.elliptic_integral(h=self.__h)
        
        eeff = 1 + .5*(self.__er-1) * Kk2/Kkp2 * Kkp1/Kk1
        return eeff
        
    def g(self):
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


    ######## WAVE PROPERTIES

    def resonant_freq(self):
        num_len = 360 / self.__elen
        Ll = self.total_inductance_per_length()
        Cl = self.capacitance_per_length()
        return 1 / (num_len*self.__l*np.sqrt(np.array(Ll)*np.array(Cl)))

    def wavelength(self,medium='cpw'):
        if medium == 'freespace':
            vp = spc.c/np.sqrt(self.__er)
            l = vp / self.resonant_freq()
        elif medium == 'effective':
            vp = spc.c/np.sqrt(self.__eeff)
            l = vp / self.resonant_freq()
        elif medium == 'cpw':
            l = self.phase_velocity() / self.resonant_freq()
        return l
    
    def phase_velocity(self):
        if self.__t == 0:
            Ll = self.geometric_inductance_per_length()
        elif self.__t > 0:
            Ll = self.total_inductance_per_length()
        Cl = self.capacitance_per_length()
        return 1 / np.sqrt(Ll*Cl)

    def phase_constant(self):
        if self.__t == 0:
            Ll = self.geometric_inductance_per_length()
        elif self.__t > 0:
            Ll = self.total_inductance_per_length()
        Cl = self.capacitance_per_length()
        return self.__fo * np.sqrt(Ll*Cl)


    ######## ELECTRICAL PROPERTIES

    def Lk(self):
        Lk = (spc.mu_0 * ((self.__pen_depth**2)
                /(self.__t*self.__w)) * self.g())
        return Lk
    
    def total_inductance_per_length(self):
        return self.Lk() + self.geometric_inductance_per_length()
        
    def geometric_inductance_per_length(self):
        Kk,Kkp = self.elliptic_integral()
        return (spc.mu_0/4) * Kkp / Kk

    def capacitance_per_length(self):
        Kk,Kkp = self.elliptic_integral()
        return 4*spc.epsilon_0*self.__eeff*(Kk / Kkp)

    def impedance_geometric(self):
        Kk,Kkp = self.elliptic_integral()
        return ( ( 30 * np.pi ) / np.sqrt(self.__eeff) ) * (Kkp / Kk)

    def impedance_kinetic(self):
        return np.sqrt(self.total_inductance_per_length() / self.capacitance_per_length())


    ######## GEOMETRY PROPERTIES

    def alpha(self,tan_d=0.005):
        eeff = self.effective_permittivity()
        ad = (self.__er/np.sqrt(eeff)) * ((eeff-1)/(self.__er-1)) * (np.pi/self.wavelength()) * tan_d
        return ad

    def beta(self,freq):
        Ll = self.total_inductance_per_length()
        Cl = self.capacitance_per_length()
        return 2*np.pi*freq*np.sqrt(Ll*Cl)

    def gamma(self,freq,tan_d=0.005):
        alpha = self.alpha(tan_d)
        beta = self.beta(freq)
        return alpha + 1j*beta