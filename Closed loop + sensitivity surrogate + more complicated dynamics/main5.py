from gp_refinement5 import RefineModel, RefineSamples
import numpy as np
import matplotlib.pyplot as plt
import math

#Define actual model
def actual(x):
    #return 19*(x-0.45)**3 + 2 - 3*(x-0.1)**2 

    return 0.6*(3 + 171/10 * x**3 - 19/10 * x**2 - 1/6 * (9*x-1)**3 + 1/120 * (9*x-1)**5 - 1/5040 * (9*x-1)**7 + 1/362880 * (9*x-1)**9 - 1/39916800 * (9*x-1)**11 + 1/6227020800 * (9*x-1)**13 - 1/1307674368000 * (9*x-1)**15 )

#derivative (used in LQR controller)
def dactual(x):
    #return 57*(x-0.45)**2 - 6*(x-0.1)

    return 0.6*(171/10 * 3 * x**2 - 19/10 * 2 * x - 1/2 * 3 * (9*x-1)**2 + 1/24 * 5 * (9*x-1)**4 - 1/720 * 7 * (9*x-1)**6 + 1/40320 * 9 * (9*x-1)**8 - 1/3628800 * 11 * (9*x-1)**10 + 1/479001600 * 13 * (9*x-1)**12 - 1/87178291200 * 15 * (9*x-1)**14 )



#List of current points that are refined in model (initial GP will be built around these points)
current_points = np.array([[0.55]])

#Range within which to search for optimal next sample
samplerange = [0, 1]


#Number of samples to refine
num_refinements = 5
#This is how many samples we want to refine at


#Refine the model
[error_norm, time] = RefineSamples(actual,dactual,current_points,samplerange,num_refinements)








