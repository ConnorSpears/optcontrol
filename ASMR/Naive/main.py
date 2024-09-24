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


#Number of samples to refine
num_refinements = 5
#This is how many samples we want to refine at


#Refine the model
#[error_norm1, error_norm2,time1,time2] = RefineSamples(actual,current_points,samples,num_refinements)



errorlist = []

for i in range(1,num_refinements+1):
    #Refine the model
    m = RefineSamples(actual,current_points,samples,i)
    errorlist.append(m)
    
    

# for i in range(0,num_refinements):
#     plt.plot(i+1,errorlist[i][0],'bo')
#     plt.plot(i+1,errorlist[i][1],'ko')

# plt.legend(['iterative','AAO'])
# plt.title("Error Norms for Greedy Refinement vs. All-at-once Refinement")
# plt.xlabel("Number of Refinements")
# plt.ylabel("Error Norm")
# plt.show()



for i in range(0,num_refinements):
    plt.plot(i+1,errorlist[i][0],'bo')
    #plt.plot(i+1,errorlist[i][1],'ko')

plt.title("Error for Naive Approach")
plt.xlabel("Number of Refinements")
plt.ylabel("Error")
plt.show()

print(errorlist)