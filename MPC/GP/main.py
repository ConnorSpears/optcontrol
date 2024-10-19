from gpMPC import GPMPC
import numpy as np
import matplotlib.pyplot as plt


#Define actual dynamics
def actual_dynamics(x): #[x1,x2,u]
    #noise = np.random.normal(0, 0.00001,2)
    x1 = x[0]
    x2 = x[1]
    u = x[2]
    #return [np.cos(u)+0.98*x1*x2+noise[0], np.sin(u)+0.01*x1+noise[1]] #[x1d,x2d]
    return [np.cos(u)+0.93*x1*x2+0.05*x1+0.03+0.05*np.sin(2*x1), 0.97*np.sin(u)+0.06*x1-0.02*x2+0.04] #[x1d,x2d]

#Define nominal dynamics
def nominal_dynamics(x): #[x1,x2,u]
    x1 = x[0]
    x2 = x[1]
    u = x[2]
    return [np.cos(u)+x1*x2, np.sin(u)] #[x1d,x2d]


GPMPC(actual_dynamics,nominal_dynamics)
    
    