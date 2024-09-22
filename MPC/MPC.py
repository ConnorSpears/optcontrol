import scipy as scipy
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.core.base.set import RangeSet
from pyomo.dae import *
import math
import numpy as np
import seaborn as sns
import sys
from typing import List


def SolveModel(func, initx1: float, initx2: float, initu:float, t_final: float, steps: int):
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, t_final))

    model.x1 = Var(model.t)
    model.x2 = Var(model.t)
    model.u = Var(model.t)

    model.x1d = DerivativeVar(model.x1, wrt=model.t)
    model.x2d = DerivativeVar(model.x2, wrt=model.t)

    # Dynamics
    def _ode_rule1(model, t):
        return model.x1d[t] == pyo.cos(model.u[t]) + func(model.x1[t]) * model.x2[t]

    def _ode_rule2(model, t):
        return model.x2d[t] == pyo.sin(model.u[t])

    model.ode1 = Constraint(model.t, rule=_ode_rule1)
    model.ode2 = Constraint(model.t, rule=_ode_rule2)

    model.obj = Objective(expr=-model.x1[t_final], sense=pyo.minimize)

    # Discretize model using finite difference
    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(model, nfe=steps, wrt=model.t)

    model.constraints = ConstraintList()

    # Initial conditions (t=0)
    model.constraints.add(expr=model.x1[0] == initx1)
    model.constraints.add(expr=model.x2[0] == initx2)

    model.constraints.add(expr=model.u[0] == initu)

    # Add terminal condition for x2
    model.constraints.add(expr=model.x2[t_final] == 0)

    # Input limits
    for t in model.t:
        model.constraints.add(expr=model.u[t] <= math.pi)
        model.constraints.add(expr=model.u[t] >= -math.pi)

    # Solve the model
    results = pyo.SolverFactory('ipopt').solve(model)
    
    return model


def propagate(x1: float, x2: float, u: float, func, dt: float,noise: bool):
    # Calculate derivatives
    dx1 = math.cos(u) + func(x1) * x2
    dx2 = math.sin(u)

    # Euler method
    x1_new = x1 + dx1 * dt
    x2_new = x2 + dx2 * dt

    # Add noise 
    if noise:
        x1_new += np.random.normal(0, 0.00013)
        x2_new += np.random.normal(0, 0.00013)

    return x1_new, x2_new

def line_search(theta, grad, xdata, udata, dts, alpha_init=10.0, rho=0.8, c=1e-4):
    """
    Backtracking line search to find optimal step size alpha.
    """
    alpha = alpha_init
    loss_initial = calculate_loss(theta, xdata, udata, dts)
    
    while True:
        # Update theta with the current alpha
        theta_new = theta - alpha * grad
        loss_new = calculate_loss(theta_new, xdata, udata, dts)
        
        # Check Armijo condition
        if loss_new <= loss_initial - c * alpha * np.abs(grad)**2:
            break
        else:
            alpha *= rho  # Reduce step size
    
    return alpha

#import random
def calculate_loss(theta, xdata, udata, dts):
    """
    Calculate the loss for a given theta.
    """
    x_pred = [xdata[0]]  # Initialize with initial state
    #if udata[0] == 0:
    #    udata[0] = udata[1]  # Replace 0 control input with the next value
    
    # Propagate for current theta value
    for j in range(len(xdata)-1):
        x1_pred, x2_pred = propagate(x_pred[j][0], x_pred[j][1], udata[j], lambda x: theta * x, dts[j], False)
        x_pred.append([x1_pred, x2_pred])
    
    # Convert predictions to numpy arrays
    x_pred = np.array(x_pred)

    # #plot
    # if random.random() < 0.0001:
    #     plt.plot([x[0] for x in xdata], [x[1] for x in xdata])
    #     plt.plot([x[0] for x in x_pred], [x[1] for x in x_pred])
    #     plt.legend(["True","Predicted"])
    #     plt.show()
    
    # Calculate loss
    loss = np.sum((x_pred - xdata) ** 2)
    
    return loss

def fit_surrogate(xdata: List[List[float]], udata: List[float], theta_init: float, dts: List[float]):
    """
    Fit surrogate model to data using gradient descent with backtracking line search.
    """
    xdata = np.array(xdata)
    udata = np.array(udata)
    dts = np.array([dt for sublist in dts for dt in sublist])  # Flatten dts

    # Truncate to final 10 data points
    h = 10
    xdata = xdata[-h:]
    udata = udata[-h:]
    dts = dts[-h:]


    #print(udata)

    # Initial guess for theta
    theta = theta_init
    max_iter = 5000
    tol = 1e-8

    loss=[]


    for i in range(max_iter):
        # if i % 1000 == 1:
        #     print(f"Iteration {i}: theta = {theta}")
        #     print(f"Loss: {calculate_loss(theta, xdata, udata, dts)}")

        # Calculate left and right loss for finite difference gradient
        theta_left = theta - 0.0001
        theta_right = theta + 0.0001

        # Compute left and right predictions
        left_loss = calculate_loss(theta_left, xdata, udata, dts)
        right_loss = calculate_loss(theta_right, xdata, udata, dts)

        # Compute gradient
        grad = (right_loss - left_loss) / 0.0002

        # Perform line search to find the best step size alpha
        alpha = line_search(theta, grad, xdata, udata, dts)


        # Update theta
        theta -= alpha * grad

        #store loss
        loss.append(calculate_loss(theta, xdata, udata, dts))

        # Check for convergence
        if np.abs(grad) < tol:
            break
    
    print(f"Iterations {i}: theta = {theta}")



    #plt.plot(loss)
    #plt.show()

    return theta






# MPC
theta=1.0

#true_theta_func = lambda x: 1.0 + 0.1 * x #true theta as a function of x1

#true_theta_func = lambda x: 1.0 if x < 0.5 else 1.0 - 0.2 * x 


x1 = 0.0
x2 = 0.0
xdata = []
udata = []

theta_pred = [[x1, theta]]
theta_real = []

dt_vals = []

t_final = 1.0
init_u = 0.0

while t_final > 0:

    print("theta: ", theta)

    #surrogate
    surrogate = lambda x: theta*x

    print(f"Current Time Horizon: {t_final}")

    # Update steps for discretization
    steps = max(int(100 * t_final), 5)
    print(f"Discretization Steps: {steps}")

    # Run optimal control
    model = SolveModel(surrogate, x1, x2, init_u, t_final, steps)

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

    for j in range(horizon):

        #true function
        #true_theta = true_theta_func(x1)
        true_theta = (t_final-dt*j)/2 + 0.5
        theta_real.append([x1,true_theta])
        true_func = lambda x: true_theta*x
        x1, x2 = propagate(x1, x2, utrunc[j], true_func, dt, True)
        xvals.append([x1,x2])
    xdata.append(xvals)

    # Update time
    t_final -= dt * horizon

    # Ensure t_final doesn't become negative
    if t_final < 1e-6:
        break

    # Fit surrogate model to data (nonlinear regression)
    # min_theta sum ||x(theta) - xdata||^2

    xflat = [x for sublist in xdata for x in sublist]
    uflat = [u for sublist in udata for u in sublist]

    theta = fit_surrogate(xflat, uflat,theta,dt_vals)

    theta_pred.append([x1,theta])
    




import matplotlib.cm as cm


fig, axs = plt.subplots(2, 1, sharex=True)

# Plot the xdata
colors = cm.rainbow(np.linspace(0, 1, len(xdata)))
for idx, xlist in enumerate(xdata):
    axs[0].plot([x[0] for x in xlist], [x[1] for x in xlist], color=colors[idx])
axs[0].set(ylabel="x2")
axs[0].set_title("State Trajectory")

# Plot the udata
for idx, ulist in enumerate(udata):
    ulist.append(ulist[-1])
    print(ulist)
    axs[1].plot([x[0] for x in xdata[idx]], [u for u in ulist], color=colors[idx])
axs[1].set(xlabel="x1", ylabel="u")
axs[1].set_title("Control Trajectory")

plt.show()






# Plot the predicted theta values
plt.plot([theta_pred[i][0] for i in range(len(theta_pred))], [theta_pred[i][1] for i in range(len(theta_pred))])
plt.plot([theta_real[i][0] for i in range(len(theta_real))], [theta_real[i][1] for i in range(len(theta_real))])
plt.legend(["Predicted Theta","True Theta"])
plt.title("True vs. Predicted Theta")
plt.xlabel("x1")
plt.ylabel("Theta")
plt.show()

