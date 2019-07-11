import numpy as np
import pandas as pd
from ResonatorOptimizer.cpwcalcs import cpwcalcs 

class Optimizer:

	def __init__(self):
		self.__keyind = lambda X,X_array: min(enumerate(X_array), key=lambda x: abs(x[1]-X))

	def set_geometry(self,swept_params,param,param_val,wavelength=1):
	    """ Sets the width, gap, and length of the cpw based on a desired input parameter"""
	    parameter = swept_params[param]
	    
	    ind, val = self.__keyind(param_val,parameter)
	    width = swept_params.width[ind]
	    gap = swept_params.gap[ind]
	    length = wavelength*(swept_params.vp[ind] / swept_params.res_freq[ind]) 

	    dic = {'width':width,'gap':gap,'length':length}
	    df = pd.DataFrame(data=dic,index=[0])
	    
	    return df