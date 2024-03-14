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



def system_dynamics(w, t, uref, K, x1ref, x2ref, actual, dactual,v):
    x1, x2 = w
    u = uref + v

    #add constraints on u

    dx1dt = np.cos(u) + actual(x1) * x2
    dx2dt = np.sin(u)
    return [dx1dt, dx2dt]


def TrackReferenceLQR(actual, dactual, x1ref, x2ref, x1dref, x2dref, uref, udref):
    # Placeholder lists for tracking states
    x1track = []
    x2track = []
    vlist = []


    # System parameters (example values, adjust as necessary)
    Q = np.eye(2) # State weighting matrix
    R = 5 # Control input weighting scalar

    # Define the time vector for integration, adjust the step size as necessary
    N = len(x1ref)
    dt = 1/N 
    t = np.linspace(0, 1, N)

    x1track.append(0)
    x2track.append(0)

    #First time step has no extra control input
    v = 0
    K = np.array([[0,0],[0,0]])

    #First propagate the dynamics forward one time step 
    #Get previous state
    w0 = [x1track[-1], x2track[-1]]

    # Integrate for one time step
    tspan = [t[0], t[0] + dt]
    w = odeint(system_dynamics, w0, tspan, args=(uref[1], K, x1ref[0], x2ref[0], actual, dactual,v))

    # Update the next initial condition
    x1track.append(w[-1][0]) 
    x2track.append(w[-1][1])

    for i in range(1, N):

        # Linearize the dynamics such that ed = A*e + B*v, where e = x-xref, v = u-uref
        A = np.array([[dactual(x1ref[i]) * x1dref[i] * x2ref[i], actual(x1ref[i])], [0, 0]])
        B = np.array([[-np.sin(uref[i]) * udref[i]], [np.cos(uref[i]) * udref[i]]])



        # Solve Riccati Equation to get optimal matrix P
        P = solve_continuous_are(A, B, Q, R)
        # Compute the optimal gain K (assuming R is scalar and B is 2x1)
        K = np.dot(B.T, P) / R


        #Get control input
        x1,x2 = w0
        e = np.array([[x1 - x1ref[i]],[x2 - x2ref[i]]])
        v = -np.dot(K, e)
        v = v[0][0]
        vlist.append(v)


        #Get previous state
        w0 = [x1track[-1], x2track[-1]]

        # Integrate for one time step
        tspan = [t[i], t[i] + dt]
        w = odeint(system_dynamics, w0, tspan, args=(uref[i], K, x1ref[i], x2ref[i], actual, dactual,v))


        # Update the next initial condition
        x1track.append(w[-1][0]) 
        x2track.append(w[-1][1])

    # plt.plot(x1ref, 'r--', label='x1ref')
    # plt.plot(x1track, 'r', label='x1track')
    # plt.show()
    # plt.plot(x2ref, 'b--', label='x2ref')
    # plt.plot(x2track, 'b', label='x2track')
    # plt.show()
        

        
    return x1track, x2track, v




