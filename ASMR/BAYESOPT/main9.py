from gp_refinement9 import RefineModel, RefineSamples
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
current_points = np.array([[0.15]])

#List of potential new points to refine
samplerange= [0,1]
#samples = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


#Number of samples to refine
num_refinements = 5
#This is how many samples we want to refine at


#Refine the model
#[error_norm, time] = RefineSamples(actual,dactual,current_points,samplerange,num_refinements)


errorlist = []


for i in range(1,num_refinements+1):
    #Refine the model
    [m, time] = RefineSamples(actual,dactual,current_points,samplerange,i)
    errorlist.append(m)

print(errorlist)
    