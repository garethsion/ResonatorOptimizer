cpwcalcs
========

This package contains the methods for performing Shwartz-Christofell mapping 
functions, necessary for determining the transmission line properties of a 
CPW resonator

conformalmapping
^^^^^^^^^^^^^^^^
.. automodule:: CPWBragg.cpwcalcs.conformalmapping
   :members:

::


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

cpw
^^^
.. automodule:: CPWBragg.cpwcalcs.cpw
   :members:


::

    #!/usr/bin/env
    import numpy as np
    import pandas as pd
    import scipy.constants as spc 
    from scipy.special import ellipk
    from ResonatorOptimizer.cpwcalcs import conformalmapping as cm

    class CPW:
        """ cpw contains the methods necessary for calculating certain parameters of 
        interest of a superconducting cpw structure. Solutions for the resonant frequency, 
        characteristic impedance, phase constant, etc, are determined by solving the
        cpw geometry analytically through conformal mapping.
        """
        def __init__(self,width=0,gap=0,length=0,elen=180,fo=0,er=0,h=None,t=0,pen_depth=None):
            """
            Constructor method - Initializes the cpw geometry

            :type width: float
            :param width: conductor width

            :type gap: float
            :param gap: gap between conductor and ground plane

            :type length: float
            :param length: conductor length
            
            :type elen: float
            :param elen: conductor electrical length (degrees)

            :type fo: float
            :param fo: designed resonant frequency

            :type er: float
            :param er: relative permittivity of substrate

            :type h: float
            :param h: thickness of substrate

            :type t: float
            :param t: thickness of conductor thin film

            :type pen_depth: float
            :param pen_depth: magnetic penetration depth
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

            self.__cm = cm.ConformalMapping(width=self.__w,
                    gap=self.__s,er=self.__er,h=self.__h,t=self.__t)
            
            if not self.__h:
                self.__eeff = (er + 1) /2
            elif self.__h:
                self.__eeff = self.__cm.effective_permittivity()

            # print('CPW with electrical length = ' + str(elen) + ' degrees')

        ######## PRINTING
        def print_cpw_params(self):
            """ returns the geometric parameters of the cpw structure.
            """
            dic = {'width':self.__w, 'gap':self.__s, 'length':self.__l,
            'h':self.__h, 't':self.__t, 'er': self.__er, 'eeff':self.__eeff,
            'pen_depth':self.__pen_depth}

            df = pd.DataFrame(data=[dic])

            return df

        def print_wave_params(self):
            """ prints out the transmission wave parameters """
            dic = {
            'fo':self.resonant_freq(),
            'wavelength':self.wavelength(),
            'vp':self.phase_velocity(),
            'phase_const':self.phase_constant()
            }

            return pd.DataFrame(data=[dic])

        def print_electrical_params(self):
            """ prints out the transmission electrical parameters """
            dic = {
            'kinetic_inductance_per_length':self.kinetic_inductance_per_length(),
            'Ltotal':self.total_inductance_per_length(),
            'Ll':self.geometric_inductance_per_length(),
            'Cl':self.capacitance_per_length(),
            'Z':self.impedance_geometric(),
            'Zki':self.impedance_total(),
            }

            return pd.DataFrame(data=[dic])

        def resonant_freq(self):
            """ Calculates the resonant frequency of the CPW """
            num_len = 360 / self.__elen
            Ll = self.total_inductance_per_length()
            Cl = self.capacitance_per_length()
            return 1 / (num_len*self.__l*np.sqrt(np.array(Ll)*np.array(Cl)))

        def wavelength(self,medium='cpw'):
            """ Calculates the wavelength of the cpw 
                :type medium: str
                :param medium: material for calculating phase velocity (e.g. freespace, cpw)
            """
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
            """ Calculates the phase velocity """
            if self.__t == 0:
                Ll = self.geometric_inductance_per_length()
            elif self.__t > 0:
                Ll = self.total_inductance_per_length()
            Cl = self.capacitance_per_length()
            return 1 / np.sqrt(Ll*Cl)

        def phase_constant(self):
            """ Calculates the phase constant """
            if self.__t == 0:
                Ll = self.geometric_inductance_per_length()
            elif self.__t > 0:
                Ll = self.total_inductance_per_length()
            Cl = self.capacitance_per_length()
            return self.__fo * np.sqrt(Ll*Cl)

        def kinetic_inductance_per_length(self):
            """ Calculates the kinetic inductance per unit length """
            kinetic_inductance_per_length = (spc.mu_0 * ((self.__pen_depth**2)
                    /(self.__t*self.__w)) * self.__cm.g())
            return kinetic_inductance_per_length
        
        def total_inductance_per_length(self):
            """ Calculates the total inductance per unity length (Lk + Lg) """
            return self.kinetic_inductance_per_length() + self.geometric_inductance_per_length()
            
        def geometric_inductance_per_length(self):
            """ Calculates the geometric inductance per unit length """
            Kk,Kkp = self.__cm.elliptic_integral()
            return (spc.mu_0/4) * Kkp / Kk

        def capacitance_per_length(self):
            """ Calculates the capacitance per unit length """
            Kk,Kkp = self.__cm.elliptic_integral()
            return 4*spc.epsilon_0*(self.__eeff*(Kk / Kkp))

        def impedance_geometric(self):
            """ Calculates the impedance, only considering the geometric contribution 
            of the inductance """
            Kk,Kkp = self.__cm.elliptic_integral()
            return ( ( 30 * np.pi ) / np.sqrt(self.__eeff) ) * (Kkp / Kk)

        def impedance_total(self):
            """ Calculates the impedance considering the total inductance """
            return np.sqrt(self.total_inductance_per_length() / self.capacitance_per_length())
