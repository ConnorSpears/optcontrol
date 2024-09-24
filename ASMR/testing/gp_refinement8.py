
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

from GP_functions8 import *
from open_loop8 import *
from LQOCP8 import *
from closed_loop8 import *



def RefineSamples(surrogate,dsurrogate,X1, samples,num_refinements):

  start_time = time.time()

  all_errors = []
  all_obj = []
  all_solerr = []
  all_polys = []
  all_samples = []
  refined_points_over_time = []
  x1tracklist = []
  x2tracklist = []

  for i in range(num_refinements):

    all_samples.append(samples)
    refined_points_over_time.append(X1)

    m,s,obj,solerr,poly,u,ud,x1d,x2d = RefineModel(surrogate,X1, samples)

    #Closed loop (ref. tracking w/ linearized LQR controller)
  
    x1openloop = obj[0]
    x2openloop = obj[1]

    x1track,x2track,v = TrackReferenceLQR(surrogate,dsurrogate,x1openloop,x2openloop,x1d,x2d,u,ud)
    x1tracklist.append(x1track)
    x2tracklist.append(x2track)




    print("New refinement point: ", samples[m])
    X1 = np.append(X1, [[samples[m]]], axis=0)
    samples = np.delete(samples, m)
    all_errors.append(s)
    
    all_obj.append(obj)
    all_solerr.append(solerr)
    all_polys.append(poly)

  
  



    
  #Need to run again to update surrogate, and get new open loop

  g = lambda x: (surrogate(x)).flatten()

  # Compute the posterior mean and covariance of the GP
  n2 = 75  # Number of points in posterior (test points)
  ny = 5  # Number of functions that will be sampled from the posterior
  domain = (-0.2, 1.2)

  #Calculate current refined points 
  y1 = g(X1)
  #Predict points at uniform spacing to capture function
  X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)

  # Compute posterior mean and covariance
  μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)

  #Function to evaluate the GP mean at a point
  def mu(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
    return μ_new
  
  #Function to evaluate the GP standard deviation at a point
  def sigma(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
    σ_new = np.sqrt(np.diag(Σ_new))
    return σ_new

  #Interpolate the GP mean using a polynomial so that the surrogate can interface with Pyomo
  GP_values = []
  x_values = list(np.arange(0, 1.001, 0.001))
  # Compute GP approximate values for 1000 points in the domain
  for i in x_values:
    mu(i)
    GP_values.append(mu(i)[0])
  #Interpolate the GP mean using a polynomial of degree 5
  p = np.polyfit(x_values,GP_values,5)
  poly = lambda x: np.polyval(p, x)
  pd = np.polyder(p)
  polyderi = lambda x: np.polyval(pd,x)

  #Solve optimal control problem using the actual and approximate models and compare the results
  modelApprox = SolveModel(surrogate)
  modelActual = SolveModel(poly)

  obj = [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()]


  all_errors.append(s)
    
  all_obj.append(obj)
  all_solerr.append(solerr)
  all_polys.append(poly)



  #Closed loop (ref. tracking w/ linearized LQR controller)
  x1openloop = obj[0]
  x2openloop = obj[1]

  x1track,x2track,v = TrackReferenceLQR(surrogate,dsurrogate,x1openloop,x2openloop,x1d,x2d,u,ud)
  x1tracklist.append(x1track)
  x2tracklist.append(x2track)




  
  #get times
  end_time = time.time()

  elapsed_time = end_time - start_time
  print(f"Iterative refinement took {elapsed_time} seconds")
  time1 = elapsed_time


  plt.rc('font', size=7)
  plt.rc('legend', fontsize=5) 
  plt.rc('axes', labelsize=7)    
  plt.rc('xtick', labelsize=7)    
  plt.rc('ytick', labelsize=7) 



  fig, ax = plt.subplots(num_refinements+1, 3, figsize=(8, 8))
  for i, sensitivity_functions in enumerate(all_obj):

    
    Actual_x1 = sensitivity_functions[0]
    Actual_x2 = sensitivity_functions[1]
    Approx_x1 = sensitivity_functions[2]
    Approx_x2 = sensitivity_functions[3]

    ax[i][0].plot(Actual_x1,Actual_x2)
    ax[i][0].plot(Approx_x1,Approx_x2)
    ax[i][0].set_xlabel("x1(t)")
    ax[i][0].set_ylabel("x2(t)")
    if (i<num_refinements):
      ax[i][0].plot(x1tracklist[i-1],x2tracklist[i-1])
      ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) vs. LQR-tracked trajectory (x_track) {i+1}")
      ax[i][0].legend(['x_c','x*','x_track'])
      ax[i][1].plot(all_samples[i],all_solerr[i],'k.')
      ax[i][1].set_xlabel("Sample points")
      ax[i][1].set_ylabel("Objective value")
      ax[i][1].set_title(f"Solution to the LQOCP for each sample {i+1}")
    else:
      ax[i][0].plot(x1tracklist[i-1],x2tracklist[i-1])
      ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) vs. LQR-tracked trajectory (x_track) {i+1}")
      ax[i][0].legend(['x_c','x*','x_track'])


    xvals = list(np.arange(0, 1.001, 0.001))
    if (i<num_refinements):
      ax[i][2].plot(xvals, all_polys[i])
    else:
      ax[i][2].plot(xvals, poly(xvals))
    ax[i][2].plot(xvals, [surrogate(xi) for xi in xvals])
    ax[i][2].set_xlabel('x1')
    ax[i][2].set_ylabel('g(x1)')
    ax[i][2].set_title(f"Surrogate model {i+1}")
    ax[i][2].legend(['Polyfit GP mean','Actual g'])
    if (i<num_refinements):
      ax[i][2].plot(refined_points_over_time[i], surrogate(refined_points_over_time[i]), 'k.', linewidth=0.1)
    else:
      ax[i][2].plot(X1, surrogate(X1), 'k.', linewidth=0.1)
      
    



  plt.tight_layout()
  plt.show()

  




  #Calculate post-refinement error

 
    
  Actual_x1 = all_obj[-1][0]
  Actual_x2 = all_obj[-1][1]
  Approx_x1 = all_obj[-1][2]
  Approx_x2 = all_obj[-1][3]

  errorx1 = np.linalg.norm(np.array(Actual_x1)-np.array(Approx_x1))
  errorx2 = np.linalg.norm(np.array(Actual_x2)-np.array(Approx_x2))

  error_norm1 = np.linalg.norm([errorx1,errorx2])

  print("Post-refinement error for iterative refinements: ", error_norm1)





  return [error_norm1 ,time1]
































def RefineModel(surrogate,X1, samples) -> float:

  global plt

  g = lambda x: (surrogate(x)).flatten()

  # Compute the posterior mean and covariance of the GP
  n2 = 75  # Number of points in posterior (test points)
  ny = 5  # Number of functions that will be sampled from the posterior
  domain = (-0.2, 1.2)

  #Calculate current refined points 
  y1 = g(X1)
  #Predict points at uniform spacing to capture function
  X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)

  # Compute posterior mean and covariance
  μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)

  # Compute the standard deviation at the test points to be plotted
  σ2 = np.sqrt(np.diag(Σ2))
  # Draw some samples of the posterior
  y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)

  # # Plot the posterior distribution and some samples
  # fig, (ax1, ax2) = plt.subplots(
  #     nrows=2, ncols=1, figsize=(6, 6))

  # #Plot the distribution of the function (mean, covariance)
  # ax1.plot(X2, g(X2), 'b--', label='$g(x)$')
  # ax1.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='black',
  #                 alpha=0.15, label='$\mu \pm 2\sigma$')
  # ax1.plot(X2, μ2, lw=2, color='black', label='$\mu$')
  # ax1.plot(X1, y1, 'ko', linewidth=2, label='$(X_i, Y_i)$')
  # ax1.set_xlabel('$x$', fontsize=13)
  # ax1.set_ylabel('$y$', fontsize=13)
  # ax1.set_title('Distribution of posterior given data (current surrogate model)')
  # ax1.axis([domain[0], domain[1], 0.2, 2.2])
  # ax1.legend()

  # # Plot some samples from this function
  # ax2.plot(X2, y2.T, '-')
  # ax2.plot(X2, g(X2), 'b--', label='$g(x)$')
  # ax2.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='black',
  #                 alpha=0.15, label='$\mu \pm 2\sigma$')
  # ax2.plot(X2, μ2, lw=2, color='black', label='$\mu$')
  # ax2.plot(X1, y1, 'ko', linewidth=2, label='$(X_i, Y_i)$')
  # ax2.set_xlabel('$x$', fontsize=13)
  # ax2.set_ylabel('$y$', fontsize=13)
  # ax2.set_title('5 different function realizations from posterior')
  # ax2.axis([domain[0], domain[1], 0, 2.2])
  # plt.tight_layout()
  # plt.show()




  #Function to evaluate the GP mean at a point
  def mu(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
    return μ_new
  
  #Function to evaluate the GP standard deviation at a point
  def sigma(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
    σ_new = np.sqrt(np.diag(Σ_new))
    return σ_new

  #Interpolate the GP mean using a polynomial so that the surrogate can interface with Pyomo
  GP_values = []
  x_values = list(np.arange(0, 1.001, 0.001))
  # Compute GP approximate values for 1000 points in the domain
  for i in x_values:
    mu(i)
    GP_values.append(mu(i)[0])
  #Interpolate the GP mean using a polynomial of degree 5
  p = np.polyfit(x_values,GP_values,5)
  poly = lambda x: np.polyval(p, x)
  pd = np.polyder(p)
  polyderi = lambda x: np.polyval(pd,x)
  #Plot the GP mean, the polynomial interpolation of the GP mean, and the actual surrogate model
  # plt.plot(x_values,GP_values,'-',linewidth=3.0)
  # plt.plot(x_values,poly(x_values))
  # gfun2 = lambda x: [surrogate(xi) for xi in x]
  # plt.plot(x_values,gfun2(x_values))
  # plt.xlabel('x1')
  # plt.ylabel('g(x1)')
  # plt.legend(['GP mean','Polyfit GP mean','Actual g'])
  # plt.title("GP mean vs. Polynomial interpolation of GP mean vs. Actual surrogate model")
  # plt.show()



  #Solve optimal control problem using the actual and approximate models and compare the results
  modelApprox = SolveModel(surrogate)
  modelActual = SolveModel(poly)

  # plt.plot(modelActual.x1[:](),modelActual.x2[:]())
  # plt.plot(modelApprox.x1[:](),modelApprox.x2[:]())
  # plt.title("Optimal trajectory (x*) vs. Current trajectory (x_c)")
  # plt.xlabel("x1(t)")
  # plt.ylabel("x2(t)")
  # plt.legend(['x*','x_c'])
  # plt.show()

  x1 = modelApprox.x1[:]()
  x2 = modelApprox.x2[:]()
  x1d = modelApprox.x1d[:]()
  x2d = modelApprox.x2d[:]()
  u = modelActual.u[:]()
  ud = modelActual.ud[:]()

  # plt.plot(modelActual.x2d[:]())
  # plt.show()
  # plt.plot(modelActual.x2d[:]())
  # plt.show()



  #Solve minmax problem to find optimal sample point to refine model at

  #Initialize list to store the objective function value for each sample point
  solution_error_estimate = []
  #Iterate through each sample point
  import matplotlib.pyplot as plt

  sensitivity_functions = []

  #samples = [samplerange[0],(samplerange[0]+samplerange[1])/2,samplerange[1]]

  for sample in samples:
    #Use 100 points at uniform spacing to capture function
    X2 = np.linspace(0, 1, 101).reshape(-1, 1)
    #Add sample to list of current refined points
    X1new = np.append(X1, [[sample]], axis=0)
    # Compute posterior covariance matrix (without evaluating the surrogate model at the new point)
    Σ2 = VariancePosterior(X1new, X2, exponentiated_quadratic)
    # Compute the standard deviation at the test points
    σ2 = np.sqrt(np.diag(Σ2))

    #Run LQOCP optimization problem
    #Pass in sample point, state variables from the open loop trajectory, and the polynomial interpolation of the surrogate model
    model_out = LQOCP(sample,x1,x2,x1d,x2d,σ2,poly,polyderi)
    
    solution_error_estimate.append(model_out.obj())
    #Plot sensitivity function
    #plt.plot(np.arange(0, 1.01, 0.01), model_out.delta[:]())
    sensitivity_functions.append(model_out.s[:]())




  # #Build gaussian process model for the sensitivity functions and perform Bayesian optimization to find the optimal sample point

  # samples =   np.array([[samplerange[0]],[(samplerange[0]+samplerange[1])/2],[samplerange[1]]])

  # #Predict points at uniform spacing to capture function
  # eval_pts = np.linspace(samplerange[0], samplerange[1], 100).reshape(-1, 1)

  # # Compute posterior mean and covariance
  # μ2, Σ2 = GP(samples, solution_error_estimate, eval_pts, exponentiated_quadratic)

  # #Function to evaluate the GP mean at a point
  # def mu2(x):
  #   X_new = np.array([[x]])  # New x-value for prediction
  #   μ_new,Σ_new = GP(samples, solution_error_estimate,X_new, exponentiated_quadratic)
  #   return μ_new
  
  # #Function to evaluate the GP standard deviation at a point
  # def sigma2(x):
  #   X_new = np.array([[x]])  # New x-value for prediction
  #   μ_new,Σ_new = GP(samples, solution_error_estimate, X_new, exponentiated_quadratic)
  #   σ_new = np.sqrt(np.diag(Σ_new))
  #   return σ_new
  

  # # Plot the posterior distribution and some samples
  # fig, (ax1, ax2) = plt.subplots(
  #     nrows=1, ncols=1, figsize=(6, 6))

  # #Plot the distribution of the function (mean, covariance)
  # ax1.plot(samples, solution_error_estimate, 'b--', label='$g(x)$')
  # ax1.fill_between(X2.flat, μ2-2*σ2, μ2+2*σ2, color='black',
  #                 alpha=0.15, label='$\mu \pm 2\sigma$')
  # ax1.plot(eval_pts, μ2, lw=2, color='black', label='$\mu$')
  # ax1.plot(X1, y1, 'ko', linewidth=2, label='$(X_i, Y_i)$')
  # ax1.set_xlabel('$x$', fontsize=13)
  # ax1.set_ylabel('$y$', fontsize=13)
  # ax1.set_title('test')
  # ax1.axis([domain[0], domain[1], 0.2, 2.2])
  # ax1.legend()

  # plt.tight_layout()
  # plt.show()


  









  

























  # plt.xlabel("t")
  # plt.ylabel("delta(t)")
  # plt.legend(['0', '0.2', '0.4', '0.6', '0.8', '1'])
  # plt.title("Delta functions for each sample point")
  # plt.show()

  #Extract the sample point that minimizes the objective function value
  sol = np.argmin(solution_error_estimate)

  # plt.plot(samples,solution_error_estimate)
  # plt.xlabel("Sample points")
  # plt.ylabel("Objective function value")
  # plt.title("Solution to the LQOCP for each sample point")

  # plt.show()

  return sol, solution_error_estimate[sol], [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()], solution_error_estimate,poly(x_values),u,ud,modelActual.x1d[:](),modelActual.x2d[:]()






