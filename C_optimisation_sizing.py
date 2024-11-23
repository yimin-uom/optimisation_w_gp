import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

maximum_loss = 0.8
sum_scale    = 1
end_p_scale  = 100
mid_p_scale  = 100
cap          = 'Cap'
floor        = 'Floor'

def opt_single_stage(data, n_trails, n_iv, grate, crate, df, prices, mannual, initial):
  #
  sv = dict()      # key (s)
  ev = dict()      # key (s, i)
  for s in data.index: 
      sv[s]= data.loc[s]['i_cost']
      for i in range(n_trails):
          try:
              ev[(s, i)] = df.loc[s,str(i)]
          except:
              ev[(s, i)] = 0
  # update contrl with mannual and initial
  initial['min'] = 0.0
  initial['max'] = 0.0
  for s in initial.index:
    if sv[s] < mannual[floor].loc[s]:
      initial.loc[s, 'max'] = sum_scale
      initial.loc[s, 'min'] = initial.loc[s, 'mannual']
    elif sv[s] > mannual[cap].loc[s]:
      initial.loc[s, 'max'] = 0
      initial.loc[s, 'min'] = 0
    else:
      initial.loc[s, 'max'] = initial.loc[s, 'mannual']
      initial.loc[s, 'min'] = initial.loc[s, 'mannual']
  try:
      maximum_loss = 0.8
      print('Optimisation beginning:')
      # Create a new model
      m = gp.Model("Energy Optimisation Problem")

      # sets
      s_s = data.index.to_list()
      s_i = range(n_trails)
      s_k = range(n_iv+3)

      # Create variables
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

def opt_second_stage(data, n_trails, n_iv, grate, crate, df, prices, mannual, initial, i):
  sv = dict()      # key (s)
  ev = dict()      # key (s, i)
  for s in data.index: 
      sv[s]= data.loc[s]['i_cost']
      for i in range(n_trails):
          try:
              ev[(s, i)] = df.loc[s,str(i)]
          except:
              ev[(s, i)] = 0
  # update contrl with mannual and initial
  # second stage updating sv for random prices
  initial['min'] = 0.0
  initial['max'] = 0.0
  for s in initial.index:
    sv[s] = prices['value'].loc[s, i, 0]
    if sv[s] < mannual[floor].loc[s]:
      initial.loc[s, 'max'] = sum_scale
      initial.loc[s, 'min'] = initial.loc[s, 'mannual']
    elif sv[s] > mannual[cap].loc[s]:
      initial.loc[s, 'max'] = 0
      initial.loc[s, 'min'] = 0
    else:
      initial.loc[s, 'max'] = initial.loc[s, 'mannual']
      initial.loc[s, 'min'] = initial.loc[s, 'mannual']

  # Create a new model
  m = gp.Model("Energy Optimisation Problem")

  # sets
  s_s = data.index.to_list()
  s_k = range(1, n_iv+3)

  # Create variables
  v_t_c   = m.addVar(name='energy_stored', lb=0)
  v_t_i   = m.addVars(s_s, name='energy_in', lb=0)
  v_p_sf  = m.addVar(name='shortfall', lb=0)
  v_p_sp = m.addVar(name='shortfall_punish', lb=0)
  v_k_sp = m.addVars(s_k, name='intra_shortfall_punish', lb=0)

  # Set objective
  m.setObjective(v_p_sf * end_p_scale + v_p_sp * mid_p_scale, GRB.MINIMIZE)

  # Add constraint:
  m.addConstr(quicksum(v_t_i[s] for s in s_s) + v_t_c == 1, "c0")

  # Add constraints multiple:
  m.addConstr(v_t_c * pow(1+crate, 2+n_iv) + quicksum(v_t_i[s]/sv[s]*ev[(s, i)] for s in s_s) + v_p_sf >= pow(1+grate, 2+n_iv))

  m.addConstr(v_p_sf - v_p_sp <= pow(1+grate, 2+n_iv) - pow(1+crate, 2+n_iv))

  m.addConstrs(quicksum(v_t_i[s] * prices['value'].loc[s, i, k]/sv[s] for s in s_s) + v_t_c * pow(1+crate, k) + v_k_sp[k] >= pow(1+crate, k) * maximum_loss for k in s_k)

  m.addConstrs(v_t_i[s] <= initial['max'].loc[s] for s in s_s)

  m.addConstrs(v_t_i[s] >= initial['min'].loc[s] for s in s_s)

  # Optimize model
  m.optimize()

  obj = m.ObjVal

  # print(f"{v_t_c.VarName} {v_t_c.X:g}")
  # for s in s_s:
  #   print(f"{v_t_i[s].VarName} {v_t_i[s].X:g}")

  # print(f"Obj: {m.ObjVal:g}")

  # Delete the model
  m.dispose()

  return obj

def opt_multi_stage(data, n_trails, n_iv, grate, crate, df, prices, mannual, initial, obj_1, obj_2, relax_1, relax_2, second_stage_scenarios):
  sv = dict()      # key (s)
  ev = dict()      # key (s, i)
  for s in data.index: 
      sv[s]= data.loc[s]['i_cost']
      for i in range(n_trails):
          try:
              ev[(s, i)] = df.loc[s,str(i)]
          except:
              ev[(s, i)] = 0
  # update contrl with mannual and initial
  initial['min_1'] = 0.0
  initial['max_1'] = 0.0
  initial['min_2'] = 0.0
  initial['max_2'] = 0.0
  for s in initial.index:
    if sv[s] < mannual[floor].loc[s]:
      initial.loc[s, 'max_1'] = sum_scale
      initial.loc[s, 'min_1'] = initial.loc[s, 'mannual']
    elif sv[s] > mannual[cap].loc[s]:
      initial.loc[s, 'max_1'] = 0
      initial.loc[s, 'min_1'] = 0
    else:
      initial.loc[s, 'max_1'] = initial.loc[s, 'mannual']
      initial.loc[s, 'min_1'] = initial.loc[s, 'mannual']
    if prices['value'].loc[s, i, 0] < mannual[floor].loc[s]:
      initial.loc[s, 'max_2'] = sum_scale
      initial.loc[s, 'min_2'] = initial.loc[s, 'mannual']
    elif prices['value'].loc[s, i, 0] > mannual[cap].loc[s]:
      initial.loc[s, 'max_2'] = 0
      initial.loc[s, 'min_2'] = 0
    else:
      initial.loc[s, 'max_2'] = initial.loc[s, 'mannual']
      initial.loc[s, 'min_2'] = initial.loc[s, 'mannual']

  try:
      maximum_loss = 0.8
      print('Optimisation beginning:')
      # Create a new model
      m = gp.Model("Energy Optimisation Problem")

      # sets
      s_s = data.index.to_list()
      s_i = range(n_trails)
      s_k = range(n_iv+3)

      # Create variables
      v_t_c_1     = m.addVar(name='energy_stored', lb=0)
      v_t_i_1   = m.addVars(s_s, name='energy_in_1', lb=0)
      v_p_sf_1  = m.addVars(s_i, name='shortfall_1', lb=0)
      v_p_sp_1  = m.addVars(s_i, name='shortfall_punish_1', lb=0)
      v_k_sp_1  = m.addVars(s_i, s_k, name='intra_shortfall_punish_1', lb=0)
      v_t_c_2     = m.addVar(name='energy_stored', lb=0)
      v_t_i_2   = m.addVars(s_s, name='energy_in_2', lb=0)
      v_p_sf_2  = m.addVars(s_i, name='shortfall_2', lb=0)
      v_p_sp_2  = m.addVars(s_i, name='shortfall_punish_2', lb=0)
      v_k_sp_2  = m.addVars(s_i, s_k, name='intra_shortfall_punish_2', lb=0)
      temp_1    = m.addVar(name='stage_1_obj', lb=0)
      temp_2    = m.addVars(s_i, name='stage_2_obj', lb=0)
      stage_1p  = m.addVar(name='stage_1_punishment', lb=-sum_scale)
      stage_2p  = m.addVar(name='stage_2_punishment', lb=-sum_scale)

      # print(obj_1)
      # print(relax_1)
      # print( obj_1 * relax_1)
      # exit()

      # Set objective
      m.setObjective(quicksum((v_t_i_1[s]/sv[s] - v_t_i_2[s]/prices['value'].loc[s, i, 0]) * (sv[s] - prices['value'].loc[s, i, 0]) for s in s_s for i in s_i) + (stage_1p + stage_2p) * end_p_scale, GRB.MINIMIZE)

      # Add constraints based on previous optimisations:
      m.addConstr(quicksum(v_p_sf_1[i] + v_p_sp_1[i] * end_p_scale for i in s_i) + quicksum(v_k_sp_1[i, k] for k in s_k for i in s_i) * mid_p_scale <= temp_1)
      m.addConstr(temp_1 - stage_1p <= obj_1 * relax_1)
      
      for j in second_stage_scenarios:
          m.addConstr(v_p_sf_2[j] + v_p_sp_2[j] * end_p_scale + quicksum(v_k_sp_2[j, k] for k in s_k) * mid_p_scale <= temp_2[j])
          m.addConstr(temp_2[j] - stage_2p <= 0, name=f"stage_2_constraint_{j}")  # obj_2[j] * relax_2 + 0.01 obj_2 does not seem to be useful, using 0 for now

      # Add constraint stage 1:     
      m.addConstr(quicksum(v_t_i_1[s] for s in s_s) + v_t_c_1 == 1)

      m.addConstrs(v_t_c_1 * pow(1+crate, 2+n_iv) + quicksum(v_t_i_1[s]/sv[s]*ev[(s, i)] for s in s_s) + v_p_sf_1[i] >= pow(1+grate, 2+n_iv) for i in s_i)

      m.addConstrs(v_p_sf_1[i] - v_p_sp_1[i] <= pow(1+grate, 2+n_iv) - pow(1+crate, 2+n_iv) for i in s_i)

      m.addConstrs(quicksum(v_t_i_1[s] * prices['value'].loc[s, i, k]/sv[s] for s in s_s) + v_t_c_1 * pow(1+crate, k) + v_k_sp_1[i, k] >= pow(1+crate, k) * maximum_loss for k in s_k for i in s_i)

      m.addConstrs(v_t_i_1[s] <= initial['max_1'].loc[s] for s in s_s)

      m.addConstrs(v_t_i_1[s] >= initial['min_1'].loc[s] for s in s_s)

      # Add constraint stage 2:     
      m.addConstr(quicksum(v_t_i_2[s] for s in s_s) + v_t_c_2 == 1)

      m.addConstrs(v_t_c_2 * pow(1+crate, 2+n_iv) + quicksum(v_t_i_2[s]/sv[s]*ev[(s, i)] for s in s_s) + v_p_sf_2[i] >= pow(1+grate, 2+n_iv) for i in s_i)

      m.addConstrs(v_p_sf_2[i] - v_p_sp_2[i] <= pow(1+grate, 2+n_iv) - pow(1+crate, 2+n_iv) for i in s_i)

      m.addConstrs(quicksum(v_t_i_2[s] * prices['value'].loc[s, i, k]/sv[s] for s in s_s) + v_t_c_2 * pow(1+crate, k) + v_k_sp_2[i, k] >= pow(1+crate, k) * maximum_loss for k in s_k for i in s_i)

      m.addConstrs(v_t_i_2[s] <= initial['max_2'].loc[s] for s in s_s)

      m.addConstrs(v_t_i_2[s] >= initial['min_2'].loc[s] for s in s_s)

      # Optimize model
      m.optimize()
      
      print(f"{v_t_c_1.VarName} {v_t_c_1.X:g}")
      for s in s_s:
        print(f"{v_t_i_1[s].VarName} {v_t_i_1[s].X:g}")
      print(f"{temp_1.VarName} {temp_1.X:g}")
      # for j in second_stage_scenarios:
      #   print(f"{temp_2[j].VarName} {temp_2[j].X:g}")

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
