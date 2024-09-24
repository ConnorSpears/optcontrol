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






def SolveModel(func):


  model = ConcreteModel()
  model.t = ContinuousSet(bounds=(0, 1))

  model.x1 = Var(model.t)

  model.x2 = Var(model.t)

  model.u = Var(model.t)


  model.x1d = DerivativeVar(model.x1, wrt=model.t)

  model.x2d = DerivativeVar(model.x2, wrt=model.t)

  model.ud = DerivativeVar(model.u, wrt=model.t)


  #dynamics
  def _ode_rule1(model, t):
    return model.x1d[t] == pyo.cos(model.u[t]) + func(model.x1[t])*model.x2[t]


  #dynamics
  def _ode_rule2(model, t):
    return model.x2d[t] ==  pyo.sin(model.u[t]) + 0.5*model.x1[t] * (model.u[t] - model.x2[t]) - (1-model.x1[t])**2 + model.x1[t]*func(model.x1[t])


  model.ode1 = Constraint(model.t, rule=_ode_rule1)
  model.ode2 = Constraint(model.t, rule=_ode_rule2)


  model.out = pyo.Objective(expr = -model.x1[1], sense=pyo.minimize)

  
  #Discretize model using Radau Collocation
  #discretizer = TransformationFactory('dae.collocation')
  #discretizer.apply_to(model,nfe=10,ncp=3,scheme='LAGRANGE-RADAU')
  discretizer = TransformationFactory('dae.finite_difference')
  discretizer.apply_to(model,nfe=100, wrt=model.t)


  N = len(list(model.t))


  model.constraints = ConstraintList()


  #Initial conditions (t=0)
  model.constraints.add(expr = model.x1[0]==0)
  model.constraints.add(expr = model.x2[0]==0)
  model.constraints.add(expr = model.x2[1]==0)

  #limits
  for i in model.t:
    model.constraints.add(expr = model.u[i] <= math.pi)
    model.constraints.add(expr = model.u[i] >= -math.pi)

    #added these constraints to prevent weird stuff happening at t=0
    model.constraints.add(expr = model.ud[i] <= 100)
    model.constraints.add(expr = model.ud[i] >= -100)
    
    
  #model.constraints.add(expr = model.u[0]==0)


  results = pyo.SolverFactory('ipopt').solve(model)
  #results.write()




  return (model)