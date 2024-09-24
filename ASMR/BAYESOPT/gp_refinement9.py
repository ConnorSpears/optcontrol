
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

from GP_functions9 import *
from open_loop9 import *
from LQOCP9 import *
from closed_loop9 import *



def RefineSamples(surrogate,dsurrogate,X1, samplerange,num_refinements):


  start_time = time.time()

  all_errors = []
  all_obj = []
  all_solerr = []
  all_polys = []
  all_samples = []
  refined_points_over_time = []
  x1tracklist = []
  x2tracklist = []
  #temp factors used to scale GP for visualization
  scale_factors = []

  #mean and var for Gp of each sensitivty function
  μ2list = []
  σ2list = []

  list_of_refined_samples = []

  for i in range(num_refinements):


    #all_samples.append(samples)
    refined_points_over_time.append(X1)

    optimal_sample, samples, obj,solerr,poly,u,ud,x1d,x2d,scale_factor,μ2, σ2 = RefineModel(surrogate,X1, samplerange)

    #Closed loop (ref. tracking w/ linearized LQR controller)
  
    x1openloop = obj[0]
    x2openloop = obj[1]

    x1track,x2track,v = TrackReferenceLQR(surrogate,dsurrogate,x1openloop,x2openloop,x1d,x2d,u,ud)
    x1tracklist.append(x1track)
    x2tracklist.append(x2track)
    scale_factors.append(scale_factor)
    μ2list.append(μ2)
    σ2list.append(σ2)




    print("New refinement point: ", optimal_sample)
    X1 = np.append(X1, [[optimal_sample]], axis=0)
    
    #all_errors.append(s)
    all_samples.append(samples)
    
    all_obj.append(obj)
    all_solerr.append(solerr)
    all_polys.append(poly)
    list_of_refined_samples.append(optimal_sample)

  
  



    
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
  polyfit = np.polyfit(x_values,GP_values,5)
  poly = lambda x: np.polyval(polyfit, x)
  pderiv = np.polyder(polyfit)
  polyderi = lambda x: np.polyval(pderiv,x)

  #Solve optimal control problem using the actual and approximate models and compare the results
  modelApprox = SolveModel(surrogate)
  modelActual = SolveModel(poly)

  obj = [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()]


  #all_errors.append(s)
    
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
      #ax[i][0].plot(x1tracklist[i-1],x2tracklist[i-1])
      #ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) vs. LQR-tracked trajectory (x_track) {i+1}")
      ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) {i+1}")
      #ax[i][0].legend(['x_c','x*','x_track'])
      ax[i][0].legend(['x_c','x*'])
      ax[i][1].plot(all_samples[i],all_solerr[i],'k.')
      evalpts = np.linspace(0,1, 75).reshape(-1, 1)
      #plot gps
      ax[i][1].fill_between(evalpts.flat, μ2list[i]-scale_factors[i]*σ2list[i], μ2list[i]+scale_factors[i]*σ2list[i], color='black',
                      alpha=0.15, label='$\mu \pm k\sigma$')
      ax[i][1].plot(evalpts, μ2list[i], lw=2, color='black', label='$\mu$')

      ax[i][1].set_xlabel("Sample points")
      ax[i][1].set_ylabel("Objective value")
      ax[i][1].set_title(f"Solution to the LQOCP for each sample {i+1}")
      ax[i][1].legend(['Evaluated refinement points','GP','Refine_point'])
    else:
      #ax[i][0].plot(x1tracklist[i-1],x2tracklist[i-1])
      #ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) vs. LQR-tracked trajectory (x_track) {i+1}")
      ax[i][0].set_title(f"Optimal trajectory (x*) vs. Current trajectory (x_c) {i+1}")
      #ax[i][0].legend(['x_c','x*','x_track'])
      ax[i][0].legend(['x_c','x*'])


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
































def RefineModel(surrogate,X1, samplerange) -> float:

  global plt

  #boundary to avoid conditioning issues
  KERNEL_BOUNDARY = 0.003

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
  polyfit = np.polyfit(x_values,GP_values,5)
  poly = lambda x: np.polyval(polyfit, x)
  pderiv = np.polyder(polyfit)
  polyderi = lambda x: np.polyval(pderiv,x)
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
  
  #store surrogate to be returned
  polyout = poly(x_values)


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
  # plt.plot(modelActual.u[:]())
  # plt.show()



  #Solve minmax problem to find optimal sample point to refine model at

  #Initialize list to store the objective function value for each sample point
  solution_error_estimate = []
  #Iterate through each sample point
  import matplotlib.pyplot as plt

  sensitivity_functions = []


  #pick 4 samples to evaluate

  a = samplerange[0]
  b = samplerange[1]

  num_samps = 6

  samples = np.linspace(a,b,num_samps)

  for i in range(num_samps):
    dist = 100
    min_dist = 100
    for ref_point in X1:
      dist = samples[i]-ref_point[0]
      if np.abs(dist) < KERNEL_BOUNDARY:
        min_dist = dist

    if min_dist != 100:
      if samples[i]==a:
        samples[i] += 1.5*KERNEL_BOUNDARY
      elif samples[i]==b:
        samples[i] -= 1.5*KERNEL_BOUNDARY
      else:
        if dist > 0:
          samples[i] += KERNEL_BOUNDARY
        else:
          samples[i] -= KERNEL_BOUNDARY



   #********************************************************************************************************************
   #Need to make sure sample points are sufficiently far away from existing refinement points to avoid conditioning issues
   #We already do this for the optimal refinement point, but it need sto be done here too
  #********************************************************************************************************************
  #Temporary fix: shifting sample points that are too close to existing refinement points

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

  
  solution_error_estimate = np.array(solution_error_estimate)


  #now construct a Gaussian process and perform Bayesian optimization to find the optimal sample point

  samples = np.array(samples).reshape(-1,1)


  nsamp = 75  # Number of points in posterior (test points)

  #Predict points at uniform spacing to capture function
  evalpts = np.linspace(a,b, nsamp).reshape(-1, 1)


  # Compute posterior mean and covariance
  μ2, Σ2 = GP(samples, solution_error_estimate, evalpts, exponentiated_quadratic)
 
  

  # Compute the standard deviation at the test points to be plotted
  σ2 = np.sqrt(np.diag(Σ2))

  # Plot the posterior distribution and some samples

  #Plot the distribution of the function (mean, covariance)



  # plt.fill_between(evalpts.flat, μ2-scale_factor*σ2, μ2+scale_factor*σ2, color='black',
  #                 alpha=0.15, label='$\mu \pm k\sigma$')
  # plt.plot(evalpts, μ2, lw=2, color='black', label='$\mu$')
  # plt.plot(samples, solution_error_estimate, 'ko', linewidth=2, label='$(X_i, Y_i)$')

  # plt.title("GP fitted to sensitivity function")
  
  # plt.tight_layout()
  # plt.show()




  #Function to evaluate the GP mean at a point
  def mu(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(samples, solution_error_estimate, X_new, exponentiated_quadratic)
    return μ_new
  
  #Function to evaluate the GP standard deviation at a point
  def sigma(x):
    X_new = np.array([[x]])  # New x-value for prediction
    μ_new,Σ_new = GP(samples, solution_error_estimate, X_new, exponentiated_quadratic)
    σ_new = np.sqrt(np.diag(Σ_new))
    return σ_new
  

  #Min expected value
  optimal_sample = evalpts[np.argmin(μ2)][0]

  # #UCB
  beta = 1
  scale_factor= (np.nanmax(μ2)-np.nanmin(μ2))/np.nanmax(σ2)
  optimal_sample = evalpts[np.argmin(μ2 - beta*scale_factor*σ2)][0]

  On = True
  iter = 0
  while On:

    #Instead of aborting if the optimal sample point is too close to an existing sample point, we should just shift it left or right
    #Fix this
    #********************************************************************************************************************


    for ref_point in samples:
      if np.abs(optimal_sample - ref_point[0]) < KERNEL_BOUNDARY:
        On = False

    for ref_point in X1:
      if np.abs(optimal_sample - ref_point[0]) < KERNEL_BOUNDARY:
        On = False

    if On:
      
      #were safe to update GP

      print("Optimal sample point: ", optimal_sample)

      #refine point at current optimal sample
      samples = np.append(samples,optimal_sample).reshape(-1,1)

      #Use 100 points at uniform spacing to capture function
      X2 = np.linspace(0, 1, 101).reshape(-1, 1)
      #Add sample to list of current refined points
      X1new = np.append(X1, [[optimal_sample]], axis=0)
  
      # Compute posterior covariance matrix (without evaluating the surrogate model at the new point)
      Σ2 = VariancePosterior(X1new, X2, exponentiated_quadratic)
      # Compute the standard deviation at the test points
      σ2 = np.sqrt(np.diag(Σ2))

      #Run LQOCP optimization problem
      #Pass in sample point, state variables from the open loop trajectory, and the polynomial interpolation of the surrogate model
      model_out = LQOCP(sample,x1,x2,x1d,x2d,σ2,poly,polyderi)
      
      solution_error_estimate = np.append(solution_error_estimate,model_out.obj())
      
      #Plot sensitivity function
      #plt.plot(np.arange(0, 1.01, 0.01), model_out.delta[:]())
      sensitivity_functions.append(model_out.s[:]())

      #now construct a Gaussian process and perform Bayesian optimization to find the optimal sample point
      # Compute posterior mean and covariance
      μ2, Σ2 = GP(samples, solution_error_estimate, evalpts, exponentiated_quadratic)
      # Compute the standard deviation at the test points to be plotted
      σ2 = np.sqrt(np.diag(Σ2))

      #Now get the new optimal sample point

      #Min expected value
      optimal_sample = evalpts[np.argmin(μ2)][0]
      #UCB
      beta = 1
      scale_factor= (np.nanmax(μ2)-np.nanmin(μ2))/np.nanmax(σ2)
      optimal_sample = evalpts[np.argmin(μ2 - beta*scale_factor*σ2)][0]




      iter += 1
      if iter > 3:
        On = False

  

  #We want to select the optimal point from our sampling
  #Min expected value
  optimal_sample = samples[np.argmin(solution_error_estimate)][0]
  #UCB





  #final scale factor for plotting
  scale_factor = (np.nanmax(μ2)-np.nanmin(μ2))/np.nanmax(σ2)


























































  #********************************************************************************************************************
  #LEAVING IT LIKE THIS FOR NOW, BUT THIS NO LONGER WORKS SINCE OUR SENSITIVITY FUNCTION IS NO LONGER CONVEX
  #********************************************************************************************************************

  #check if its within 0.05 of any existing sample point
  flag = False
  for point in X1:
    distance = optimal_sample - point[0]
    if np.abs(distance) < KERNEL_BOUNDARY:
      flag = True

  #if it is, we need to shift it left or right until its not within 0.05 of any exisiting sample, and still within bounds
  #we pick the shift direction that results in the lower function value
  if flag:
    #first left
    opt_temp = optimal_sample
    point_too_close = True
    while point_too_close:
      distance = -100
      for i in range(len(X1)):
        d = opt_temp - X1[i][0]
        if np.abs(d) < KERNEL_BOUNDARY:
          distance = d
        if (i == len(X1)-1) and distance == -100:
          point_too_close = False
      if point_too_close:
        #shift it left
        opt_temp = opt_temp - (KERNEL_BOUNDARY + distance)
    #now right
    opt_temp2= optimal_sample
    point_too_close = True
    while point_too_close:
      distance = -100
      for i in range(len(X1)):
        d = opt_temp2 - X1[i][0]
        if np.abs(d) < KERNEL_BOUNDARY:
          distance = d
        if (i == len(X1)-1) and distance == -100:
          point_too_close = False
      if point_too_close:
        #shift it right
        opt_temp2 = opt_temp2 + (KERNEL_BOUNDARY - distance)

    #make sure they are in bounds
    opt_temp = np.clip(opt_temp,0,1)
    opt_temp2 = np.clip(opt_temp2,0,1)

    #pick the one resulting in the lower function value
    f1 = mu(opt_temp)
    f2 = mu(opt_temp2)
    if f1 < f2:
      optimal_sample = opt_temp
    else:
      optimal_sample = opt_temp2

    #check if its within 0.03 of any existing sample point
    for ref_point in X1:
      if np.abs(opt_temp - ref_point[0]) < KERNEL_BOUNDARY:
        optimal_sample = opt_temp2

    for ref_point in X1:
      if np.abs(opt_temp2 - ref_point[0]) < KERNEL_BOUNDARY:
        optimal_sample = opt_temp




  #make sure optimal sample is real valued (remove imaginary component)
  optimal_sample = np.real(optimal_sample)








  

            
             
 

















  # plt.xlabel("t")
  # plt.ylabel("delta(t)")
  # plt.legend(['0', '0.2', '0.4', '0.6', '0.8', '1'])
  # plt.title("Delta functions for each sample point")
  # plt.show()

  # plt.plot(samples,solution_error_estimate)
  # plt.xlabel("Sample points")
  # plt.ylabel("Objective function value")
  # plt.title("Solution to the LQOCP for each sample point")

  # plt.show()

  return optimal_sample, samples, [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()], solution_error_estimate,polyout,u,ud,modelActual.x1d[:](),modelActual.x2d[:](),scale_factor,μ2,σ2






