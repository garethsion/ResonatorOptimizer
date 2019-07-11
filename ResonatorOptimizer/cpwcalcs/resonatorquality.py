import numpy as np

class ResonatorQuality:
    def __init__(self,freq,s21,length=1e-03,alpha=0):
        self.__freq = freq
        self.__s21 = s21
        self.__l = length
        self.__alpha = alpha
        return
    
    def Qint(self):
        return np.pi / (self.__alpha * 2 * self.__l)
    
    def Qext(self):
        return 1/(1/self.Qloaded() - 1/self.Qint())
    
    def Qloaded(self):
        rfit = ResonatorFitting()
        fit = rfit.LorentzianFit(self.__freq,self.__s21,plot=False)
        return fit.Q.values[0]
    
    def insertion_loss(self,format='db'):
        g = self.Qint()/self.Qext()
        if format == 'db':
            return -20*np.log10(g/(g+1))
        elif format == 'mag':
            return g/(g+1)
        else:
            raise ValueError('The format specified is not recognized. Please choose either \'db\' or \'mag\'')
