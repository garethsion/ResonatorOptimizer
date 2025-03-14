#!/usr/bin/env python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ResonatorOptimizer.cpwcalcs import cpw 
    
class ParamSweeps():
    """ A class for sweeping geometric parameters of a cpw
    """
    def __init__(self,length,total_width,fo,er,h,t,pen_depth):
        """
            Constructor method - Initializes the cpw geometry

            :type length: float
            :param length: conductor length

            :type total_width: float
            :param width: conductor = 2*gap width
                        
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
        self.__length = length
        self.__total_width = total_width
        self.__fo = fo
        self.__er = er
        self.__h = h
        self.__t = t
        self.__pen_depth = pen_depth
        return
    
    def width_to_gap(self,minw,maxw,wit=0.2):
        """ Calculates the transmission parameters for a given width and gap
        range

        :type minw: float
        :param minw: min conductor width

        :type maxw: float
        :param maxw: max conductor width

        :type wit: float
        :param mwit: iterator
        """
        Zcpw = []
        Zki = []
        wcpw = []
        scpw = []

        Cl = []
        Ll = []
        Lkl = []
        Ltot = []
        
        vp = []
        
        wlist = list(np.arange(minw,maxw,wit))

        for w in range(1,len(wlist)):
            width = wlist[w]
            wcpw.append(width*1e-06)
            scpw.append(.5*(self.__total_width - wcpw[w-1]))
            cp = cpw.CPW(width=wcpw[w-1],gap=scpw[w-1],length=self.__length,
                fo=self.__fo,er=self.__er,h=self.__h,t=self.__t,pen_depth=self.__pen_depth)
            Zcpw.append(cp.impedance_geometric())
            Zki.append(cp.impedance_total())

            Cl.append(cp.capacitance_per_length())
            Ll.append(cp.geometric_inductance_per_length())
            Lkl.append(cp.kinetic_inductance_per_length())
            Ltot.append(cp.total_inductance_per_length())  
            vp.append(cp.phase_velocity())
            
        res_freq = 1 / (2*self.__length*np.sqrt(np.array(Ltot)*np.array(Cl)))
            
        data = {'width':wcpw,
                'gap':scpw,
                'Z':Zcpw,
                'Zki':Zki,
                'Cl':Cl,
                'Ll':Ll,
                'Lkl':Lkl,
                'Ltot':Ltot,
                'vp':vp,
                'res_freq':res_freq
               }
        
        parameters = pd.DataFrame(data=data)
        return parameters

    def plot_params(self,params):
        """ Plot the transmission parameters

        :type params: pd dataframe
        :param params: transmission parameters
        """

        fig = plt.figure(figsize=(15,10))
        ax1 = plt.subplot(221)
        ax1.plot(params.width / params.gap, params.Z,'o',markersize=10,label='Normal')
        ax1.plot(params.width / params.gap, params.Z,linewidth=3.5,alpha=.5,label='_nolegend_')
        ax1.plot(params.width / params.gap, params.Zki,'s',markersize=10,label='Superconducting')
        ax1.plot(params.width / params.gap, params.Zki,linewidth=3.5,alpha=.5,label='_nolegend_')
        ax1.set_ylabel('Z ($\Omega$)',fontsize=28)
        ax1.legend(fontsize=15)
        ax1.grid()

        ax2 = plt.subplot(222)
        ax2.plot(params.width / params.gap, params.Ll*1e06,'o',markersize=10,label='$L_{g}$')
        ax2.plot(params.width / params.gap, params.Ll*1e06,linewidth=5.5,alpha=.5,label='_nolegend_')
        ax2.plot(params.width / params.gap, params.Lkl*1e06,'s',markersize=10,label='$L_{k}$')
        ax2.plot(params.width / params.gap, params.Lkl*1e06,linewidth=5.5,alpha=.5,label='_nolegend_') 
        ax2.plot(params.width / params.gap, params.Ltot*1e06,color='k',linewidth=5.5,alpha=.75,label='$L = L_{g} + L_{k}$') 
        ax2.set_ylabel('Inductance ($\mu$ H)',fontsize=28)
        ax2.legend(fontsize=15)
        ax2.grid()

        ax3 = plt.subplot(223)
        ax3.plot(params.width / params.gap, params.vp,'o',markersize=10)
        ax3.set_ylabel('$\\upsilon_{p}$ (m/s) ',fontsize=28)
        ax3.grid()

        ax4 = plt.subplot(224)
        ax4.plot(params.width / params.gap, params.res_freq*1e-09,'o',markersize=10)
        ax4.set_ylabel('$f_{0}$ (GHz) ',fontsize=28)
        ax4.grid()

        fig.text(0.5, 0.04, 'w/s',fontsize=28, ha='center')
        # # plt.savefig('Nb_Bragg_Z_Lk_vp_fo.eps')
        plt.show()

# fig.text(0.5, 0.04, 'w/s',fontsize=28, ha='center')
# # plt.savefig('Nb_Bragg_Z_Lk_vp_fo.eps')
# plt.show()
