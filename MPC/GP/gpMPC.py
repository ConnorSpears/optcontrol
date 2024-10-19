
import scipy as scipy
#import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.core.base.set import RangeSet
from pyomo.dae import *
from re import I
import math
import numpy as np
import seaborn as sns
import sys
from itertools import combinations
import time
sns.set_style('darkgrid')
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import matplotlib.cm as cm


from GP_functions import *
from open_loop import *




def propagate(dynamics, x1: float, x2: float, u: float, dt: float):
    out = dynamics([x1,x2,u])
    dx1 = out[0]
    dx2 = out[1]

    # Euler method
    x1_new = x1 + dx1 * dt
    x2_new = x2 + dx2 * dt

    return x1_new, x2_new

    
#Function to evaluate the GP mean at a point
def mu(point,x,y1):
    X_new = np.array([[point[0], point[1], point[2]]])  # New 3d x-value for prediction
    μ_new, Σ_new = GP(x, y1, X_new, exponentiated_quadratic)
    return μ_new

# Function to evaluate the GP standard deviation at a 2D point
def sigma(new_state,x,y1):
    X_new = np.array([[point[0], point[1], point[2]]])  # New 3d x-value for prediction
    μ_new, Σ_new = GP(x, y1, X_new, exponentiated_quadratic)
    σ_new = np.sqrt(np.diag(Σ_new))
    return σ_new



# Polynomial function that evaluates the fitted polynomial at any (x1, x2,u)
def poly_4d(x1, x2, u, coeffs, degree):
    z = 0  # Initialize as a scalar
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                z += coeffs[index] * (x1**i) * (x2**j) * (u**k)
                index += 1
    return z




# Helper functions to fit and evaluate the polynomial
def fit_4d_polynomial(x1, x2, u, z, degree):
    """ Fit a 3D polynomial surface of the given degree to data (x1, x2, u, z) """
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                terms.append((x1**i) * (x2**j) * (u**k))
    A = np.column_stack(terms)
    # Solve the least-squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs




def GPMPC(actual_dynamics, nominal_dynamics):

    GP1_error = []
    GP2_error = []


    x1 = 0.0
    x2 = 0.0
    xdata = []
    udata = []

    dt_vals = []

    t_final = 1.0
    init_u = 0.0

    #initial GP surrogate (0 function)
    GP_refinements = [] #x1,x2,u
    GP_refinements_vals_1 = [] #error in derivative of x1
    GP_refinements_vals_2 = [] #error in derivative of x2
    GP_surrogate_1 = lambda x: 0
    GP_surrogate_2 = lambda x: 0
    #
    #


    while t_final > 0:



        print(f"Current Time Horizon: {t_final}")

        # Update steps for discretization
        steps = max(int(100 * t_final), 5)
        print(f"Discretization Steps: {steps}")

        # Run optimal control
        model = SolveModel(GP_surrogate_1, GP_surrogate_2, x1, x2, init_u, t_final, steps)

        #get the predicted x values
        x1_values = [pyo.value(model.x1[t]) for t in model.t]
        x2_values = [pyo.value(model.x2[t]) for t in model.t]
        x_values = np.array([[x1_values[i], x2_values[i]] for i in range(len(x1_values))])

        # Take first few control inputs and propagate model
        u_values = [pyo.value(model.u[t]) for t in model.t]

        # Determine the control horizon (either 10 steps or fewer if near the end)
        horizon = min(10, len(u_values))
        utrunc = u_values[:horizon]
        utrunc[0]=init_u
        init_u = utrunc[-1]
        if utrunc[0] == 0:
            utrunc[0] = utrunc[1]

        #update u_data
        #udata.extend(utrunc)
        udata.append(utrunc)

        # Calculate time step
        dt = t_final / len(u_values)
        dt_vals.append([dt]*horizon)

        xvals=[]
        xvals.append([x1,x2])

  

        for j in range(horizon-1):
            #propagate actual dynamics (this simulates observing the system as it progresses)
            x1_next, x2_next = propagate(actual_dynamics,x1, x2, utrunc[j], dt)
            #estimate the derivative of the real system
            f_real = [(x1_next-x1)/dt, (x2_next-x2)/dt]
            #update x1, x2
            x1 = x1_next
            x2 = x2_next
            #get the derivative of the nominal system
            f_nom = nominal_dynamics([x1,x2,utrunc[j]])
            #append real x values
            xvals.append([x1,x2])
            #calculate error in derivative and add to list so we can refine the surrogate
            f_error = [f_real[0]-f_nom[0], f_real[1]-f_nom[1]]
            print(f"Error in derivative: {f_error}")
            GP_refinements.append([x1,x2,utrunc[j]])
            GP_refinements_vals_1.append(f_error[0]) #error in derivative of x1 to be fit by GP1
            GP_refinements_vals_2.append(f_error[1]) #error in derivative of x2 to be fit by GP2

            #keep track of how well the GP is fitting
            GP1_error.append(abs(f_error[0]-mu([x1,x2,utrunc[j]],GP_refinements[::3],GP_refinements_vals_1[::3])[0]))
            GP2_error.append(abs(f_error[1]-mu([x1,x2,utrunc[j]],GP_refinements[::3],GP_refinements_vals_2[::3])[0]))



        #append last x value
        x1, x2 = propagate(actual_dynamics,x1, x2, utrunc[-1], dt)
        xvals.append([x1,x2])

        #truncate the lists to prevent matrix conditioning issues (only take every third point)
        GP_refinements_trunc = GP_refinements[::3]
        GP_refinements_vals_1_trunc = GP_refinements_vals_1[::3]
        GP_refinements_vals_2_trunc = GP_refinements_vals_2[::3]


        #refine GP surrogates after each MPC run
        #update GP
        #μ2, Σ2 = GP(GP_refinements, GP_refinements_vals, points, exponentiated_quadratic) # points are the points that we want to evaluate in the GP model
        #Interpolate the GP mean using a polynomial so that the surrogate can interface with Pyomo
        x1_values = np.linspace(0, 1, 50)
        x2_values = np.linspace(0, 1, 50)
        u_values = np.linspace(-math.pi, math.pi, 50)
        # Compute GP approximate values for a grid of points in the domain
        x1_vals, x2_vals, u_vals = np.meshgrid(x1_values, x2_values,u_values)
        X_vals = np.column_stack([x1_vals.ravel(), x2_vals.ravel(), u_vals.ravel()])
        GP_values_1 = np.array([mu([x1,x2,u],GP_refinements_trunc,GP_refinements_vals_1_trunc)[0] for (x1, x2, u) in X_vals])
        GP_values_2 = np.array([mu([x1,x2,u],GP_refinements_trunc,GP_refinements_vals_2_trunc)[0] for (x1, x2, u) in X_vals])
        #Fit polynomial to values to pass to Pyomo
        # Perform a least-squares polynomial fit to the 4D data - We will use a 4D polynomial fit
        degree = 5
        coeffs_1 = fit_4d_polynomial(x1_vals.ravel(), x2_vals.ravel(), u_vals.ravel(), GP_values_1, degree)
        coeffs_1 = list(coeffs_1)
        coeffs_2 = fit_4d_polynomial(x1_vals.ravel(), x2_vals.ravel(), u_vals.ravel(), GP_values_2, degree)
        coeffs_2 = list(coeffs_2)
        #get the polynomial functions
        GP_surrogate_1 = lambda x: poly_4d(x[0], x[1], x[2], coeffs_1,degree)
        GP_surrogate_2 = lambda x: poly_4d(x[0], x[1], x[2], coeffs_2,degree)









        xdata.append(xvals)

        # Update time
        t_final -= dt * horizon

        # Ensure t_final doesn't become negative
        if t_final < 1e-6:
            break

    

    #plot 

    # Plot the xdata
    colors = cm.rainbow(np.linspace(0, 1, len(xdata)))
    for idx, xlist in enumerate(xdata):
        plt.plot([x[0] for x in xlist], [x[1] for x in xlist], color=colors[idx])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GPMPC')

    plt.show()





    #nominal trajectory

    gp = lambda x: 0
    x1 = 0.0
    x2 = 0.0
    init_u = 0.0
    t_final = 1.0
    steps = 100

    model = SolveModel(gp, gp, x1, x2, init_u, t_final, steps)

    #get the predicted x values
    x1_values = [pyo.value(model.x1[t]) for t in model.t]
    x2_values = [pyo.value(model.x2[t]) for t in model.t]
    x_values = np.array([[x1_values[i], x2_values[i]] for i in range(len(x1_values))])

    #u vals
    u_values = [pyo.value(model.u[t]) for t in model.t]

    # Calculate time step
    dt = t_final / len(u_values)

    xvals=[]
    xvals.append([x1,x2])


    for j in range(len(u_values)):
        #propagate actual dynamics (this simulates observing the system as it progresses)
        x1,x2 = propagate(actual_dynamics,x1, x2, u_values[j], dt)
        xvals.append([x1,x2])

    plt.figure()
    plt.plot([x[0] for x in xvals], [x[1] for x in xvals], color='black')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Nominal Trajectory (No MPC, No GP)')
    plt.show()

    plt.figure()
    plt.plot(GP1_error[:-5])
    plt.plot(GP2_error[:-5])
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.title('GP Error')
    plt.legend(['GP1 Error','GP2 Error'])
    plt.show()

    










