import scipy as scipy
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.core.base.set import RangeSet
from pyomo.dae import *
from re import I
import math
import numpy as np
import seaborn as sns
import sys
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
from scipy.integrate import solve_ivp




#SOLUTION USING ALGEBRAIC RICCATI EQUATION SOLVED AT EACH TIME STEP


# def system_dynamics(w, t, uref, K, x1ref, x2ref, actual, dactual,v):
#     x1, x2 = w
#     u = uref + v

#     dx1dt = np.cos(u) + actual(x1) * x2
#     dx2dt = np.sin(u)
#     return [dx1dt, dx2dt]


# def TrackReferenceLQR(actual, dactual, x1ref, x2ref, x1dref, x2dref, uref, udref):
#     # Placeholder lists for tracking states
#     x1track = []
#     x2track = []
#     vlist = []


#     # System parameters 
#     Q = np.eye(2) # State weighting matrix
#     R = 5 # Control input weighting scalar

#     # Define the time vector for integration, adjust the step size as necessary
#     N = len(x1ref)
#     dt = 1/N 
#     t = np.linspace(0, 1, N)

#     x1track.append(0)
#     x2track.append(0)

#     #First time step has no extra control input
#     v = 0
#     K = np.array([[0,0],[0,0]])

#     #First propagate the dynamics forward one time step 
#     #Get previous state
#     w0 = [x1track[-1], x2track[-1]]

#     # Integrate for one time step
#     tspan = [t[0], t[0] + dt]
#     w = odeint(system_dynamics, w0, tspan, args=(uref[1], K, x1ref[0], x2ref[0], actual, dactual,v))

#     # Update the next initial condition
#     x1track.append(w[-1][0]) 
#     x2track.append(w[-1][1])

#     for i in range(1, N):

#         # Linearize the dynamics such that ed = A*e + B*v, where e = x-xref, v = u-uref
#         A = np.array([[dactual(x1ref[i]) * x1dref[i] * x2ref[i], actual(x1ref[i])], [0, 0]])
#         B = np.array([[-np.sin(uref[i]) * udref[i]], [np.cos(uref[i]) * udref[i]]])



#         # Solve Riccati Equation to get optimal matrix P
#         P = solve_continuous_are(A, B, Q, R)
#         # Compute the optimal gain K (assuming R is scalar and B is 2x1)
#         K = np.dot(B.T, P) / R


#         #Get control input
#         x1,x2 = w0
#         e = np.array([[x1 - x1ref[i]],[x2 - x2ref[i]]])
#         v = -np.dot(K, e)
#         v = v[0][0]
#         vlist.append(v)


#         #Get previous state
#         w0 = [x1track[-1], x2track[-1]]

#         # Integrate for one time step
#         tspan = [t[i], t[i] + dt]
#         w = odeint(system_dynamics, w0, tspan, args=(uref[i], K, x1ref[i], x2ref[i], actual, dactual,v))


#         # Update the next initial condition
#         x1track.append(w[-1][0]) 
#         x2track.append(w[-1][1])

#     # plt.plot(x1ref, 'r--', label='x1ref')
#     # plt.plot(x1track, 'r', label='x1track')
#     # plt.show()
#     # plt.plot(x2ref, 'b--', label='x2ref')
#     # plt.plot(x2track, 'b', label='x2track')
#     # plt.show()
        

        
#     return x1track, x2track, v










#SOLUTION USING DIFFERENTIAL RICCATI EQUATION


def system_dynamics(w, t, uref, K, x1ref, x2ref, actual, dactual,v):
    x1, x2 = w
    u = uref + v

    dx1dt = np.cos(u) + actual(x1) * x2 + 1.5 * actual(x1) * (x1 * np.sin(11*x1) + 0.05)
    dx2dt = np.sin(u)
    return [dx1dt, dx2dt]



# Define the Differential Riccati Equation
def diff_riccati(t, P_flat, A_func, B_func, Q, R, x1ref, x2ref, x1dref, uref, udref, actual, dactual,dt,N):

    P = P_flat.reshape((2,2))
    
    #Evaluate A and B using the provided functions and the current time t
    
    #Convert time t to index i
    i = int(np.clip(t/dt, 0, N-1))  
    A = A_func(x1ref[i], x2ref[i], x1dref[i], actual, dactual)
    #print("u:", uref[i], "u_dot:", udref[i])
    B = B_func(uref[i], udref[i])
    
    # Calculate the derivative of P
    P_dot = -P @ A - A.T @ P - Q + P @ B @ np.linalg.inv(R) @ B.T @ P
    return P_dot.flatten()  

#A and B as functions of state and control 
def A_func(x1, x2, x1_dot, actual, dactual):
    return np.array([[dactual(x1) * x1_dot * x2 +1.5* ( dactual(x1)*x1_dot*(x1*sin(11*x1-2)+0.05) + actual(x1)*((sin(11*x1-2)) + 11*x1*cos(11*x1-2)) ), actual(x1)], [0, 0]])

def B_func(u, u_dot):
    return np.array([[-np.sin(u) * u_dot], [np.cos(u) * u_dot]])



def TrackReferenceLQR(actual, dactual, x1ref, x2ref, x1dref, x2dref, uref, udref):
    x1track = []
    x2track = []
    vlist = []

    udref[0] = udref[1]


    Q = np.eye(2)  
    R = np.array([[5]])  
    N = len(x1ref)  
    dt = 1/N  
    T = 1  
    t = np.linspace(0, 1, N)

    #initial condition for P at final time T (P(T))
    P_final = np.zeros((2,2)).flatten()

    #integrate the DRE backward in time
    sol = solve_ivp(
        diff_riccati,
        [T, 0],  
        P_final,
        args=(A_func, B_func, Q, R, x1ref, x2ref, x1dref, uref, udref, actual, dactual, dt, N),
        t_eval=np.linspace(T, 0, N)
    )


    #extract P(t) for each time step
    P_t = [sol.y[:, i].reshape((2,2)) for i in range(sol.y.shape[1])]



    #flip P_t since we integrated backward
    P_t = P_t[::-1]


    x1track.append(x1ref[0])
    x2track.append(x2ref[0])
    w0 = [x1ref[0], x2ref[0]] 


    for i in range(1, N):

        A = A_func(x1ref[i], x2ref[i], x1dref[i], actual, dactual)
        B = B_func(uref[i], udref[i])
        P = P_t[i]
        

        K = np.dot(B.T, P) / R[0][0]


        #Get control input
        x1,x2 = w0
        e = np.array([[x1 - x1ref[i]],[x2 - x2ref[i]]])
        v = -np.dot(K, e)
        v = v[0][0]
        vlist.append(v)


        #Get previous state
        w0 = [x1track[-1], x2track[-1]]

        #integrate for one time step
        tspan = [t[i], t[i] + dt]
        w = odeint(system_dynamics, w0, tspan, args=(uref[i], K, x1ref[i], x2ref[i], actual, dactual,v))


        #update the next initial condition
        x1track.append(w[-1][0]) 
        x2track.append(w[-1][1])



    return x1track, x2track, v










