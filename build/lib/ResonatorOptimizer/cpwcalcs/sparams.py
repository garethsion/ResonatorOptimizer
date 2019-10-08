import numpy as np

class Sparams:
    def __init__(self,freq=0,Z0=50,length=1e-03,gamma=1,Ck=1e-15):
        self.__freq = freq
        self.__Z0 = Z0
        self.__l = length
        self.__gamma = gamma
        self.__Ck = Ck
        return

    def port(self):
        Zport = 1 / (1j*2*np.pi*self.__freq*self.__Ck)
        return [np.matrix(((1,Zport[i]),(0,1)),dtype=complex) for i in range(len(self.__freq))]

    def transmission(self):
        t11 = np.cosh(self.__gamma*self.__l)
        t12 = self.__Z0 * np.sinh(self.__gamma*self.__l)
        t21 = (1/self.__Z0) * np.sinh(self.__gamma*self.__l)
        t22 = np.cosh(self.__gamma*self.__l)
        return [np.matrix(([t11[i],t12[i]],[t21[i],t22[i]]),dtype=complex) for i in range(len(self.__gamma))]

    def transfer(self,inport,transmission,outport):
        return inport * transmission * outport

    def s11_from_abcd(self,abcd,Rload=50,format='db'):
        s11_mat = [self.get_s11(abcd[i],Rload,format) for i in range(len(self.__freq))]
        s11 = np.array([complex(s11_mat[i]) for i in range(len(s11_mat))])
        return s11

    def s21_from_abcd(self,abcd,Rload=50,format='db'):
        s21_mat = [self.get_s21(abcd[i],Rload,format) for i in range(len(self.__freq))]
        s21 = np.array([complex(s21_mat[i]) for i in range(len(s21_mat))])
        return s21

    def s21(self,Rload=50,format='db'):
        inport = self.port()
        tr = self.transmission()
        outport = self.port()
        abcd_mat = [self.transfer(inport[i],tr[i],outport[i]) for i in range(len(self.__freq))]
        s21_mat = [self.get_s21(abcd_mat[i],Rload,format) for i in range(len(self.__freq))]
        s21 = np.array([complex(s21_mat[i]) for i in range(len(s21_mat))])
        return s21

    def get_s11(self,abcd,Rload,format):
        A = abcd.flat[0]
        B = abcd.flat[1]
        C = abcd.flat[2]
        D = abcd.flat[3]

        s11 = (A + (B/Rload) - C*Rload - D) / (A + (B/Rload) + C*Rload + D)

        if format == 'db':
            return 20*np.log10(s11)
        elif format == 'mag':
            return s11
        else:
            raise ValueError('The format specified is not recognized. Please choose either \'db\' or \'mag\'')


    def get_s21(self,abcd,Rload,format):
        A = abcd.flat[0]
        B = abcd.flat[1]
        C = abcd.flat[2]
        D = abcd.flat[3]

        s21 = 2 / (A + (B/Rload) + C*Rload + D)

        if format == 'db':
            return 20*np.log10(s21)
        elif format == 'mag':
            return s21
        else:
            raise ValueError('The format specified is not recognized. Please choose either \'db\' or \'mag\'')
