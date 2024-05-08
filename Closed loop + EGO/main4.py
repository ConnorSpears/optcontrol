from gp_refinement4 import RefineModel, RefineSamples
import numpy as np
import matplotlib.pyplot as plt


#Define actual model
def actual(x):
    #return 4 * x**2 - 3*x**3 + 1
    return 19*(x-0.45)**3 + 2 - 3*(x-0.1)**2 

#derivative (used in LQR controller)
def dactual(x):
    return 57*(x-0.45)**2 - 6*(x-0.1)

#plt.plot(np.linspace(0,1,100),actual(np.linspace(0,1,100)))
#plt.show()

#List of current points that are refined in model (initial GP will be built around these points)
current_points = np.array([[0.55]])

#Range within which to search for optimal next sample
samplerange = [0, 1]


#Number of samples to refine
num_refinements = 4
#This is how many samples we want to refine at


#Refine the model
[error_norm, time] = RefineSamples(actual,dactual,current_points,samplerange,num_refinements)

