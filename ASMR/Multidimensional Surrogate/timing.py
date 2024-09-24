from gp_refinement import RefineModel, RefineSamples
import numpy as np
import matplotlib.pyplot as plt



#Define actual model
def actual(x):
    #return 4 * x**2 - 3*x**3 + 1
    return 19*(x-0.45)**3 + 2 - 3*(x-0.1)**2 

#plt.plot(np.linspace(0,1,100),actual(np.linspace(0,1,100)))
#plt.show()

#List of current points that are refined in model (initial GP will be built around these points)
current_points = np.array([[0.25]])

#List of potential new points to refine
samples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#samples = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

#Timing comparison
timeIter = []
timeAAO = []
for i in range(1,5):
    [error_norm1, error_norm2,time1,time2] = RefineSamples(actual,current_points,samples,i)
    timeIter.append(time1)
    timeAAO.append(time2)

plt.plot(range(1,5),timeIter,'bo')
plt.plot(range(1,5),timeAAO,'ko')
plt.legend(['iterative','AAO'])
plt.title("Time for Greedy Refinement vs. All-at-once Refinement")
plt.xlabel("Number of Refinements")
plt.ylabel("Time (s)")
plt.show()