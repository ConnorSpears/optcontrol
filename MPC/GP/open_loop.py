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




def SolveModel(gp1, gp2, initx1: float, initx2: float, initu:float, t_final: float, steps: int):
  model = ConcreteModel()
  model.t = ContinuousSet(bounds=(0, t_final))

  model.x1 = Var(model.t)
  model.x2 = Var(model.t)
  model.u = Var(model.t)

  model.x1d = DerivativeVar(model.x1, wrt=model.t)
  model.x2d = DerivativeVar(model.x2, wrt=model.t)

  #dynamics
  def _ode_rule1(model, t):
    gp_prediction = gp1([model.x1[t],model.x2[t],model.u[t]]) #gaussian process learned dynamics gets added to nominal model
    return model.x1d[t] == pyo.cos(model.u[t])+ model.x1[t]*model.x2[t] + gp_prediction

  def _ode_rule2(model, t):
    gp_prediction = gp2([model.x1[t],model.x2[t],model.u[t]]) #gaussian process learned dynamics gets added to nominal model
    return model.x2d[t] ==  pyo.sin(model.u[t]) + gp_prediction

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

