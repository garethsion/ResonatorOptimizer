import ResonatorOptimizer as ro
import numpy as np
import pandas as pd

class Bragg:
    def __init__(self,fo,er,h,t,pen_depth,no_mirrors=5):
        self.__fo = fo
        self.__er = er
        self.__h = h
        self.__t = t
        self.__pen_depth = pen_depth
        self.__no_mirrors = no_mirrors
        return
    
    def get_abcd(self,cpw,freq,length):
        gamma = cpw.gamma(freq,tan_d=0.005)
        Z0 = cpw.impedance_kinetic()

        sp = ro.Sparams(freq=freq,gamma=gamma,length=length,Z0=Z0)
        abcd = sp.transmission()
        return abcd

    def cpw_section(self,width, gap, length):
        cpw = ro.cpwCalcs(width,gap,length,self.__fo, self.__er, h=self.__h, t=self.__t, pen_depth=self.__pen_depth)
        return cpw

    def bragg_resonator(self,freq,lowZ, highZ, cavity):  
        cav_abcd = self.get_abcd(cavity,freq,cavity.cpw_params().length.values[0])
        lowZ_abcd = self.get_abcd(lowZ,freq,lowZ.cpw_params().length.values[0])
        highZ_abcd = self.get_abcd(highZ,freq,highZ.cpw_params().length.values[0])
        
        mirror_lhs = [np.matmul(lowZ_abcd[i], highZ_abcd[i]) for i in range(len(freq))]
        mirror_rhs = [np.matmul(highZ_abcd[i], lowZ_abcd[i]) for i in range(len(freq))]
        bragg = [np.matmul(mirror_lhs[i]**self.__no_mirrors, np.matmul(cav_abcd[i], 
        	mirror_rhs[i]**self.__no_mirrors)) for i in range(len(freq))]

        sp = ro.Sparams(freq=freq)
        bragg_s21 = sp.s21_from_abcd(bragg)
        return bragg_s21
