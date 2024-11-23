import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

def rules(data, n_trails, n_iv, grate, crate, df, prices, mannual, initial):

  try:
      print('D Optimisation beginning:')
      # Create a new model
      m = gp.Model("Energy Optimisation Problem")

      # sets
      s_i = range(n_trails)
      s_k = range(n_iv+3)

      # Create variables
      v_rule_do = m.addVar(name='energy_stored', lb=0)
      v_rule_up = m.addVar(name='energy_stored', lb=0)
      v_t_c   = m.addVar(name='energy_stored', lb=0)
      v_t_i   = m.addVars(s_s, name='energy_in', lb=0)
      v_p_sf  = m.addVars(s_i, name='shortfall', lb=0)
      v_p_sp = m.addVars(s_i, name='shortfall_punish', lb=0)
      v_k_sp = m.addVars(s_i, s_k, name='intra_shortfall_punish', lb=0)


      # Set objective
      m.setObjective(quicksum(v_p_sf[i] + v_p_sp[i] * end_p_scale for i in s_i) + quicksum(v_k_sp[i, k] for k in s_k for i in s_i) * mid_p_scale, GRB.MINIMIZE)

      # Add constraint:
      m.addConstr(quicksum(v_t_i[s] for s in s_s) + v_t_c == 1, "c0")

      # Add constraints multiple:
      m.addConstrs(v_t_c * pow(1+crate, 2+n_iv) + quicksum(v_t_i[s]/sv[s]*ev[(s, i)] for s in s_s) + v_p_sf[i] >= pow(1+grate, 2+n_iv) for i in s_i)

      m.addConstrs(v_p_sf[i] - v_p_sp[i] <= pow(1+grate, 2+n_iv) - pow(1+crate, 2+n_iv) for i in s_i)

      m.addConstrs(quicksum(v_t_i[s] * prices['value'].loc[s, i, k]/sv[s] for s in s_s) + v_t_c * pow(1+crate, k) + v_k_sp[i, k] >= pow(1+crate, k) * maximum_loss for k in s_k for i in s_i)

      # m.addConstrs(v_t_i[s] <= initial['max'].loc[s] for s in s_s)

      # m.addConstrs(v_t_i[s] >= initial['min'].loc[s] for s in s_s)

      # Optimize model
      m.optimize()
      
      print(f"{v_t_c.VarName} {v_t_c.X:g}")
      for s in s_s:
        print(f"{v_t_i[s].VarName} {v_t_i[s].X:g}")

      print(f"Obj: {m.ObjVal:g}")

      obj = m.ObjVal

      # Delete the model
      m.dispose()

  except gp.GurobiError as e:
      print(f"Error code {e.errno}: {e}")
      exit()

  except AttributeError:
      print("Encountered an attribute error")
      exit()
  
  return obj