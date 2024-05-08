
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




def LQOCP(sample,x1,x2,x1d,x2d,σ2,poly,polyderi):

  model = ConcreteModel()
  model.t = ContinuousSet(bounds=(0, 1))

  model.s1 = Var(model.t)
  model.s2 = Var(model.t)

  model.delta = Var(model.t)

  model.s1d = DerivativeVar(model.s1, wrt=model.t)
  model.s2d = DerivativeVar(model.s2, wrt=model.t)

  def rule1(model, t):

    eval = int(t*100)

    A1 = polyderi(x1[eval])*x1d[eval]*x2[eval]
    A2 = poly(x1[eval])
    B = x2[eval]

    return model.s1d[t] == A1*model.s1[t] + A2*model.s2[t] + B*model.delta[t]
  
  def rule2(model, t):

    eval = int(t*100)

    A1 = 1/2 * (poly(x1[eval])-x2[eval]) + 2*(1-x1[eval])
    A2 = -1/2 * x1[eval]
    B = x1[eval]

    return model.s1d[t] == A1*model.s1[t] + A2*model.s2[t] + B*model.delta[t]


  model.ode1 = Constraint(model.t, rule=rule1)
  model.ode2 = Constraint(model.t, rule=rule2)



  def myintegral(model,i):
    return model.s1[i] **2 + model.s2[i] **2

  model.n = Integral(model.t, wrt=model.t, rule=myintegral)

  def myobjective(model):
     return model.n

  model.obj = Objective(rule=myobjective,sense=pyo.maximize)



  #Discretize model using Radau Collocation
  #discretizer = TransformationFactory('dae.collocation')
  #discretizer.apply_to(model,nfe=10,ncp=3,scheme='LAGRANGE-RADAU')
  discretizer = TransformationFactory('dae.finite_difference')
  discretizer.apply_to(model,nfe=100, wrt=model.t)



  N = len(list(model.t))


  model.constraints = ConstraintList()


  #Initial conditions (t=0)
  model.constraints.add(expr = model.s1[0]==0)
  model.constraints.add(expr = model.s2[0]==0)


  #limits
  k=0
  for i in model.t:
    lim = σ2[k]
    if math.isnan(lim):
      lim = 0
    model.constraints.add(expr = model.delta[i] <= lim)
    model.constraints.add(expr = model.delta[i] >= lim)
  

    k = k+1


  results = pyo.SolverFactory('ipopt').solve(model)
  #results.write()
  #if results.solver.status == 'ok':
      #model.pprint()

  return model