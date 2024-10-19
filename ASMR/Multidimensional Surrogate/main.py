from gp_refinement import RefineModel, RefineSamples
from gp_refinement_Bayes_Opt import RefineModel_BO, RefineSamples_BO
import numpy as np
import matplotlib.pyplot as plt


#Define actual model
def actual(x):
    # x is expected to be a list of 2D points, where each point is a list [x1, x2]
    # Compute actual values for each point in the list
    return 19 * (x[0] - 0.45) ** 3 + 2 - 3 * (x[0] - 0.1) ** 2 + 0.1 * x[1]




#plt.plot(np.linspace(0,1,100),actual(np.linspace(0,1,100)))
#plt.show()

#List of current points that are refined in model (initial GP will be built around these points)
current_points = np.array([[0.25,0.1],[0.75,0.1],[0.75,0.9],[0.25,0.9],[0.5,0.5],[0.1,0.9],[0.9,0.9]])

#List of potential new points to refine
samples = [[0.5,0.2],[0.25,0.2],[0.25,0.5],[0.3,0.15],[0.7,0.4],[0.1,0.1],[0.9,0.1]]


#Number of samples to refine
num_refinements = 3
#This is how many samples we want to refine at


#Refine the model
#[error_norm1, error_norm2,time1,time2] = RefineSamples(actual,current_points,samples,num_refinements)




#m = RefineSamples(actual,current_points,samples,num_refinements)
m = RefineSamples_BO(actual,current_points,num_refinements)
    
    