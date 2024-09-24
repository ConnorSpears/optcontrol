from gp_refinement7 import RefineModel, RefineSamples
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

#List of potential new points to refine
samples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#samples = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


#Number of samples to refine
num_refinements = 4
#This is how many samples we want to refine at


#Refine the model
[error_norm, time] = RefineSamples(actual,dactual,current_points,samples,num_refinements)

