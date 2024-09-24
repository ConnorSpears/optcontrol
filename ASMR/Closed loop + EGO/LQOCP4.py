
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

  model.s = Var(model.t)
  #model.s2 = Var(model.t)

  model.delta = Var(model.t)

  model.sd = DerivativeVar(model.s, wrt=model.t)

  def rule1(model, t):

    #Very hacky solution that probably should be changed
    eval = int(t*100)

    # B = x2[eval]

    # A = poly(x1[eval])*x2d[eval]

    # return model.sd[t] == A*model.s[t] + B*model.delta[t]

    B = x2[eval]

    A = polyderi(x1[eval])*x1d[eval]*x2[eval] + polyderi(x1[eval])*x2[eval]
    
    #A = polyderi(x1[eval])*x1d[eval]*x2[eval] + poly(x1[eval])*x2d[eval]

    return model.sd[t] == A*model.s[t] + B*model.delta[t]


  model.ode1 = Constraint(model.t, rule=rule1)



  def myintegral(model,i):
    #return model.s[i] **2 + model.s2[i] **2
    return model.s[i] **2 

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
  model.constraints.add(expr = model.s[0]==0)


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