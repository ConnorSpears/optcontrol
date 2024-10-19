
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


from GP_functions import *
from open_loop import *
from LQOCP import *




def RefineSamples_BO(surrogate,X1,num_refinements):

  start_time = time.time()

  #save_copies = (surrogate,X1, samples,num_refinements)

  all_errors = []
  all_obj = []
  all_solerr = []
  all_polys = []
  all_samples = []
  refined_points_over_time = []
  μ2list = []
  σ2list = []
  scale_factors = []

  for i in range(num_refinements):

    refined_points_over_time.append(X1)

    m,s,obj,solerr,poly,samples_out,μ2,σ2,scale = RefineModel_BO(surrogate,X1)
    

    X1 = np.append(X1, [samples_out[m]], axis=0)
    print("X1")
    print(X1)

    #del samples[m]
    all_errors.append(s)
    
    all_obj.append(obj)
    all_solerr.append(copy.deepcopy(solerr))
    all_polys.append(poly)

    all_samples.append(copy.deepcopy(samples_out))
    μ2list.append(copy.deepcopy(μ2))
    σ2list.append(copy.deepcopy(σ2))
    scale_factors.append(copy.deepcopy(scale))
    
  # #Need to run again to update surrogate, and get new open loop

  # g = lambda x: (surrogate(x)).flatten()

  # # Compute the posterior mean and covariance of the GP
  # n2 = 75  # Number of points in posterior (test points)
  # ny = 5  # Number of functions that will be sampled from the posterior
  # domain = (-0.2, 1.2)

  # #Calculate current refined points 
  # y1 = g(X1)
  # #Predict points at uniform spacing to capture function
  # X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)

  # # Compute posterior mean and covariance
  # μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)

  # #Function to evaluate the GP mean at a point
  # def mu(x):
  #   X_new = np.array([[x]])  # New x-value for prediction
  #   μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
  #   return μ_new
  
  # #Function to evaluate the GP standard deviation at a point
  # def sigma(x):
  #   X_new = np.array([[x]])  # New x-value for prediction
  #   μ_new,Σ_new = GP(X1, y1, X_new, exponentiated_quadratic)
  #   σ_new = np.sqrt(np.diag(Σ_new))
  #   return σ_new

  # #Interpolate the GP mean using a polynomial so that the surrogate can interface with Pyomo
  # GP_values = []
  # x_values = list(np.arange(0, 1.001, 0.001))
  # # Compute GP approximate values for 1000 points in the domain
  # for i in x_values:
  #   mu(i)
  #   GP_values.append(mu(i)[0])
  # #Interpolate the GP mean using a polynomial of degree 5
  # p = np.polyfit(x_values,GP_values,5)
  # poly = lambda x: np.polyval(p, x)
  # pd = np.polyder(p)
  # polyderi = lambda x: np.polyval(pd,x)

  # #Solve optimal control problem using the actual and approximate models and compare the results
  # modelApprox = SolveModel(surrogate)
  # modelActual = SolveModel(poly)

  # obj = [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()]

  # #solerr = solution_error_estimate[sol]

  # #X1 = np.append(X1, [[samples[m]]], axis=0)
  # #samples = np.delete(samples, m)
  # all_errors.append(s)
    
  # all_obj.append(obj)
  # all_solerr.append(solerr)
  # all_polys.append(poly)

  
  #get times
  end_time = time.time()

  elapsed_time = end_time - start_time
  print(f"Iterative refinement took {elapsed_time} seconds")
  time1 = elapsed_time


  # plt.rc('font', size=7)
  # plt.rc('legend', fontsize=5) 
  # plt.rc('axes', labelsize=7)    
  # plt.rc('xtick', labelsize=7)    
  # plt.rc('ytick', labelsize=7) 



  fig = plt.figure(figsize=(12, 12))

  # Create a 3D subplot layout
  ax = [fig.add_subplot(num_refinements+1, 3, i+1) if i % 3 == 0 else fig.add_subplot(num_refinements+1, 3, i+1, projection='3d') for i in range((num_refinements+1) * 3)]

  for i, sensitivity_functions in enumerate(all_obj):
    Actual_x1 = sensitivity_functions[0]
    Actual_x2 = sensitivity_functions[1]
    Approx_x1 = sensitivity_functions[2]
    Approx_x2 = sensitivity_functions[3]

    # Plot optimal vs. current trajectory in 2D
    ax[i * 3].plot(Actual_x1, Actual_x2, label='Current trajectory')
    ax[i * 3].plot(Approx_x1, Approx_x2, label='Optimal trajectory')
    ax[i * 3].set_title(f"Optimal vs. Current trajectory {i+1}")
    ax[i * 3].set_xlabel("x1(t)")
    ax[i * 3].set_ylabel("x2(t)")
    ax[i * 3].legend()

    if i < num_refinements:
      # Convert list to numpy array
      samples_array = np.array(all_samples[i])
      solerr = all_solerr[i]
      # Split 2D points into x and y for scatter plotting
      sample_x, sample_y = samples_array[:, 0], samples_array[:, 1]
      ax[i * 3 + 1].scatter(sample_x, sample_y, solerr, marker='.')
      ax[i * 3 + 1].set_xlabel("x1")
      ax[i * 3 + 1].set_ylabel("x2")
      ax[i * 3 + 1].set_title(f"Solution to the LQOCP for each sample {i+1}")
      # Compute the posterior mean and covariance of the GP
      n2 = 100  # Number of points in posterior (test points)
      domain = (-0.2, 1.2)
      #Predict points at uniform spacing to capture function
      x1_grid = np.linspace(domain[0], domain[1], n2)
      x2_grid = np.linspace(domain[0], domain[1], n2)
      evalpts = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])
      #plot gps
      print(len(μ2list[i]))
      print(len(σ2list[i]))
      print(len(evalpts))
      #ax[i*3+1].fill_between(evalpts, μ2list[i]-scale_factors[i]*σ2list[i], μ2list[i]+scale_factors[i]*σ2list[i], color='black',
      #                alpha=0.15, label='$\mu \pm k\sigma$')
      #ax[i*3+1].plot(evalpts, μ2list[i], lw=2, color='black', label='$\mu$')

      # Assuming domain, n2, μ2list, σ2list, and scale_factors are defined

      x1_grid = np.linspace(domain[0], domain[1], n2)
      x2_grid = np.linspace(domain[0], domain[1], n2)
      X1, X2 = np.meshgrid(x1_grid, x2_grid)

      # Reshape μ2list[i] and σ2list[i] to match the grid shape
      mu = μ2list[i].reshape(n2, n2)
      sigma = σ2list[i].reshape(n2, n2)


      # Plot the mean surface
      #ax[i*3+1].plot_surface(X1, X2, mu, color='black', alpha=0.6, label='$\mu$')

      #ax[i*3+1].scatter(x1_grid, x2_grid, mu, c='k', marker='.', linewidth=0.1)

      # Plot the uncertainty bounds
     # ax[i*3+1].plot_surface(X1, X2, mu - scale_factors[i] * sigma, color='black', alpha=0.15, label='$\mu - k\sigma$')
      #ax[i*3+1].plot_surface(X1, X2, mu + scale_factors[i] * sigma, color='black', alpha=0.15, label='$\mu + k\sigma$')









    # Create a 2D grid
    x1_vals = np.linspace(0, 1, 100)
    x2_vals = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Evaluate the surrogate on the grid
    Z_vals = np.array([surrogate(np.array([x1, x2])) for x1, x2 in X_grid])
    Z_vals = Z_vals.reshape(X1.shape)

    # Plot real model
    poly = all_polys[i]
    ax[i * 3 + 2].plot_surface(X1, X2, Z_vals, cmap='Oranges', edgecolor='none', label='Real g', alpha=0.5)

    # Evaluate the polynomial over the grid
    Z_poly = np.array([poly(np.array([x1, x2])) for x1, x2 in X_grid])
    Z_poly = Z_poly.reshape(X1.shape)

    # Plot the polynomial surface
    ax[i * 3 + 2].plot_surface(X1, X2, Z_poly, cmap='Blues', edgecolor='none', label='Surrogate model', alpha=0.5)

    if i < num_refinements:
        ax[i * 3 + 2].scatter(refined_points_over_time[i][:, 0], refined_points_over_time[i][:, 1], 
                              [surrogate(pt) for pt in refined_points_over_time[i]], c='k', marker='.', linewidth=0.1)
    else:
        ax[i * 3 + 2].scatter(X1.ravel(), X2.ravel(), 
                              [surrogate(np.array([x1, x2])) for x1, x2 in X_grid], c='k', marker='.', linewidth=0.1)
    ax[i * 3 + 2].set_xlabel('x1')
    ax[i * 3 + 2].set_ylabel('x2')
    ax[i * 3 + 2].set_zlabel('g(x1, x2)')
    ax[i * 3 + 2].set_title(f"Real g vs. Surrogate model {i + 1}")
    ax[i * 3 + 2].legend()





  plt.tight_layout()
  plt.show()
    




  #Calculate post-refinement error

 
    
  Actual_x1 = all_obj[-1][0]
  Actual_x2 = all_obj[-1][1]
  Approx_x1 = all_obj[-1][2]
  Approx_x2 = all_obj[-1][3]

  errorx1 = np.linalg.norm(np.array(Actual_x1)-np.array(Approx_x1))
  errorx2 = np.linalg.norm(np.array(Actual_x2)-np.array(Approx_x2))

  error_norm = np.linalg.norm([errorx1,errorx2])

  print("Post-refinement error for iterative refinements: ", error_norm)





  return [error_norm, time1]
































def RefineModel_BO(surrogate,x) -> float:

  global plt

  #Define actual model
  def actual(x):
    # x is expected to be a list of 2D points, where each point is a list [x1, x2]
    # Compute actual values for each point in the list
    return [19 * (point[0] - 0.45) ** 3 + 2 - 3 * (point[0] - 0.1) ** 2 + 0.1 * point[1] for point in x]

  g = lambda x: actual(x)

  # Compute the posterior mean and covariance of the GP
  n2 = 10  # Number of points in posterior (test points)
  ny = 5  # Number of functions that will be sampled from the posterior
  domain = (-0.2, 1.2)

  #Calculate current refined points 
  y1 = g(x)
  #Predict points at uniform spacing to capture function
  x1_grid = np.linspace(domain[0], domain[1], n2)
  x2_grid = np.linspace(domain[0], domain[1], n2)
  X2 = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])

  # Compute posterior mean and covariance




  μ2, Σ2 = GP(x, y1, X2, exponentiated_quadratic)

  # Compute the standard deviation at the test points to be plotted
  σ2 = np.sqrt(np.diag(Σ2))
  # Draw some samples of the posterior
  y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)




  #Function to evaluate the GP mean at a point
  def mu(x_new):
    X_new = np.array([[x_new[0], x_new[1]]])  # New 2D x-value for prediction
    μ_new, Σ_new = GP(x, y1, X_new, exponentiated_quadratic)
    return μ_new

    # Function to evaluate the GP standard deviation at a 2D point
  def sigma(x_new):
    X_new = np.array([[x_new[0], x_new[1]]])  # New 2D x-value for prediction
    μ_new, Σ_new = GP(x, y1, X_new, exponentiated_quadratic)
    σ_new = np.sqrt(np.diag(Σ_new))
    return σ_new

  #Interpolate the GP mean using a polynomial so that the surrogate can interface with Pyomo
  GP_values = []
  x1_values = np.linspace(0, 1, 50)
  x2_values = np.linspace(0, 1, 50)




  # Compute GP approximate values for a grid of points in the domain
  x1_vals, x2_vals = np.meshgrid(x1_values, x2_values)
  X_vals = np.column_stack([x1_vals.ravel(), x2_vals.ravel()])
  GP_values = np.array([mu([x1,x2])[0] for (x1, x2) in X_vals])

   # Polynomial function that evaluates the fitted polynomial at any (x1, x2)
  def poly_2d(x1, x2):
    #degree = int(np.sqrt(len(coeffs))) - 1
    z = 0  # Initialize as a scalar
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[index] * (x1**i) * (x2**j)  # Here, x1 and x2 are symbolic (Pyomo vars)
            index += 1
    return z


  def polyder_2d(x1, x2):
    # Manually compute the derivatives of poly_2d with respect to x1 and x2
    #degree = int(np.sqrt(len(coeffs))) - 1
    grad_x1 = 0
    grad_x2 = 0
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i > 0:
                grad_x1 += i * coeffs[index] * (x1**(i - 1)) * (x2**j)
            if j > 0:
                grad_x2 += j * coeffs[index] * (x1**i) * (x2**(j - 1))
            index += 1
    return [grad_x1, grad_x2]


  # Helper functions to fit and evaluate a 2D polynomial
  def fit_2d_polynomial(x1, x2, z, degree):
    """ Fit a 2D polynomial surface of the given degree to data (x1, x2, z) """
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x1**i) * (x2**j))
    A = np.column_stack(terms)
    # Build matrix of polynomial terms up to the given degree
    #A = np.column_stack([x1**i * x2**j for i in range(degree + 1) for j in range(degree + 1 - i)])
    # Solve the least-squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


  # Perform a least-squares polynomial fit to the 2D data (x1, x2, GP_values)
  # We will use a 2D polynomial fit with terms like a*x1^i*x2^j
  degree = 7
  coeffs = fit_2d_polynomial(x1_vals.ravel(), x2_vals.ravel(), GP_values, degree)
  coeffs = list(coeffs)


  poly = lambda x: poly_2d(x[0], x[1])
  polyderi = lambda x: polyder_2d(x[0], x[1])

  #Plot the GP mean, the polynomial interpolation of the GP mean, and the actual surrogate model


  # Create a grid for the plot
  x1_values_plot, x2_values_plot = np.meshgrid(x1_values, x2_values)

  # Compute GP mean values on the grid for plotting
  GP_mean_values = np.array([mu([x1, x2])[0] for (x1, x2) in zip(x1_values_plot.ravel(), x2_values_plot.ravel())])
  GP_mean_values = GP_mean_values.reshape(x1_values_plot.shape)

  # Compute polynomial interpolation values on the grid for plotting
  poly_values = np.array([poly_2d(x1, x2) for (x1, x2) in zip(x1_values_plot.ravel(), x2_values_plot.ravel())])
  poly_values = poly_values.reshape(x1_values_plot.shape)

  # # Create a 3D plot for the GP mean
  # fig = plt.figure(figsize=(10, 7))
  # ax = fig.add_subplot(111, projection='3d')
  # ax.plot_surface(x1_values_plot, x2_values_plot, GP_mean_values, cmap='viridis', alpha=0.6, label='GP Mean')
  # ax.plot_surface(x1_values_plot, x2_values_plot, poly_values, cmap='plasma', alpha=0.6, label='Polynomial Fit')

  # # Reshape GP_values to match the shape of x1_values and x2_values
  # GP_values_reshaped = GP_values.reshape(x1_values_plot.shape)
  # # Create a 3D scatter plot
  # ax.scatter(x1_values_plot, x2_values_plot, GP_values_reshaped, c='k', marker='.', linewidth=0.1)

  # # Customize plot
  # ax.set_xlabel('x1')
  # ax.set_ylabel('x2')
  # ax.set_zlabel('Function Value')
  # ax.set_title('Comparison of GP Mean and Polynomial Fit')
  # ax.legend()

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



  #Solve minmax problem to find optimal sample point to refine model at

  #Initialize list to store the objective function value for each sample point
  solution_error_estimate = []
  #Iterate through each sample point
  import matplotlib.pyplot as plt

  sensitivity_functions = []



  def create_sigma2_interpolator(x1_grid, x2_grid, sigma2_grid):
    """
    Create a 2D linear interpolator for variance values.

    Parameters:
    - x1_grid: 1D array of grid points in the x1 dimension.
    - x2_grid: 1D array of grid points in the x2 dimension.
    - sigma2_grid: 2D array of variance values corresponding to the x1_grid and x2_grid.

    Returns:
    - interpolator: A function that takes (x1, x2) and returns the interpolated variance value.
    """
    # Create the 2D interpolator
    interpolator = RegularGridInterpolator((x1_grid, x2_grid), sigma2_grid, method='linear')
    
    def sigma2_func(x1, x2):
        """
        Interpolates the variance at a given (x1, x2) point.

        Parameters:
        - x1: The x1-coordinate (within the range of x1_grid).
        - x2: The x2-coordinate (within the range of x2_grid).

        Returns:
        - Interpolated variance at (x1, x2).
        """
        # Ensure the input is a 2D point
        point = np.array([[x1, x2]])
        return interpolator(point)[0]
    
    return sigma2_func












  #BAYES OPT

  #construct a Gaussian process and perform Bayesian optimization to find the optimal sample point

  #pick 4 inital samples to build GP around
  samples = [[0.2,0.2],[0.8,0.8],[0.2,0.8],[0.8,0.2]]

  #Refine samples
  solution_error_estiamte = []
  sensitivity_functions = []
  for sample in samples:
    #Use 100 points at uniform spacing to capture function
    n2=100
    x1_grid = np.linspace(-0.2, 1.2, n2)
    x2_grid = np.linspace(-0.2, 1.2, n2)
    X2 = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])
    #Add sample to list of current refined points
    X1new = np.append(x, [sample], axis=0)
    # Compute posterior covariance matrix (without evaluating the surrogate model at the new point)
    Σ2 = VariancePosterior(X1new, X2, exponentiated_quadratic)
    # Compute the standard deviation at the test points
    σ2 = np.sqrt(np.diag(Σ2))
    #Reshape
    σ2_grid = σ2.reshape(n2, n2)
    σ2_func = create_sigma2_interpolator(x1_grid, x2_grid, σ2_grid)
    #Create a list of variance values for each sample point along the trajectory
    σ2 = []
    for a,b in zip(x1,x2):
      σ2.append(σ2_func(a,b))
    #Run LQOCP optimization problem
    #Pass in sample point, state variables from the open loop trajectory, and the polynomial interpolation of the surrogate model
    model_out = LQOCP(sample,x1,x2,x1d,x2d,σ2,poly,polyderi)
    solution_error_estimate.append(model_out.obj())
    #Plot sensitivity function
    #plt.plot(np.arange(0, 1.01, 0.01), model_out.delta[:]())
    sensitivity_functions.append(model_out.s[:]())





  n2 = 100  # Number of points in posterior (test points)

  #Predict points at uniform spacing to capture function
  #evalpts = np.linspace(a,b, nsamp).reshape(-1, 1)

  x1_grid = np.linspace(domain[0], domain[1], n2)
  x2_grid = np.linspace(domain[0], domain[1], n2)
  evalpts = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])

  # Compute posterior mean and covariance

  μ2, Σ2 = GP(np.array(samples), solution_error_estimate, evalpts, exponentiated_quadratic,{'l': 0.3, 'σp': 1})
 
  # Compute the standard deviation at the test points to be plotted
  σ2 = np.sqrt(np.diag(Σ2))

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


  #Perform BO

  #Min expected value
  optimal_sample = evalpts[np.argmin(μ2)]

  # #UCB
  # beta = 1
  # scale_factor= (np.nanmax(μ2)-np.nanmin(μ2))/np.nanmax(σ2)
  # optimal_sample = evalpts[np.argmin(μ2 - beta*scale_factor*σ2)]



  
  iter = 0

  not_converged = True

  while not_converged:

    optimal_sample_list = optimal_sample.tolist()
   

    #refine point at current optimal sample
    samples.append(optimal_sample_list)


    #Use 100 points at uniform spacing to capture function
    n2=100
    x1_grid = np.linspace(-0.2, 1.2, n2)
    x2_grid = np.linspace(-0.2, 1.2, n2)
    X2 = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])
    #Add sample to list of current refined points

   
    X1new = np.append(x, [optimal_sample_list], axis=0)
    # Compute posterior covariance matrix (without evaluating the surrogate model at the new point)
    Σ2 = VariancePosterior(X1new, X2, exponentiated_quadratic)

    # Compute the standard deviation at the test points
    σ2 = np.sqrt(np.diag(Σ2))

    #Reshape
    σ2_grid = σ2.reshape(n2, n2)

    σ2_func = create_sigma2_interpolator(x1_grid, x2_grid, σ2_grid)

    #Create a list of variance values for each sample point along the trajectory
    σ2 = []
    for a,b in zip(x1,x2):
      σ2.append(σ2_func(a,b))


    #Run LQOCP optimization problem
    #Pass in sample point, state variables from the open loop trajectory, and the polynomial interpolation of the surrogate model
    model_out = LQOCP(sample,x1,x2,x1d,x2d,σ2,poly,polyderi)
    
    solution_error_estimate.append(model_out.obj())
    #Plot sensitivity function
    #plt.plot(np.arange(0, 1.01, 0.01), model_out.delta[:]())
    sensitivity_functions.append(model_out.s[:]())


    # Compute posterior mean and covariance
    μ2, Σ2 = GP(np.array(samples), solution_error_estimate, evalpts, exponentiated_quadratic,{'l': 0.3, 'σp': 1})
    # Compute the standard deviation at the test points to be plotted
    σ2 = np.sqrt(np.diag(Σ2))

    
    #Now get the new optimal sample point

    #Min expected value
    optimal_sample = evalpts[np.argmin(μ2)]
    #UCB
    # beta = 1
    scale_factor= (np.nanmax(μ2)-np.nanmin(μ2))/np.nanmax(σ2)
    # optimal_sample = evalpts[np.argmin(μ2 - beta*scale_factor*σ2)]


    iter += 1
    if iter > 3:
      not_converged = False






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

  return sol, solution_error_estimate[sol], [modelActual.x1[:](),modelActual.x2[:](),modelApprox.x1[:](),modelApprox.x2[:]()], solution_error_estimate, poly, samples ,μ2,σ2,scale_factor




