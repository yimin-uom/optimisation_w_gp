import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.stats import halfnorm
import random

min_g   = 0.03
peg_std = 0.39

## reorganise the series to match correlation ##
def adjust_lists_to_desired_inter_correlations(lists, desired_inter_list_corrs, max_iterations, tolerance=1e-2):
    print('Adjusting random variables to match correlations!')
    n = len(lists)
    m = len(lists[0])  # Assuming all lists have the same length
    
    # Initializations
    current_lists = np.copy(lists)
    
    for iteration in range(max_iterations):
        # 1. Calculate current inter-list correlations
        current_inter_list_corrs = np.corrcoef(current_lists)
        
        # 2. Calculate errors between current and desired inter-list correlations
        errors = current_inter_list_corrs - desired_inter_list_corrs
        
        # 3. Check convergence
        max_error = np.max(np.abs(errors))
        if max_error < tolerance:
            print(f"Iteration {iteration + 1}: Converged with max error {max_error}")
            break
        
        # 4. Update current_lists based on errors
        for i in range(n):
            for j in range(m):
                # Scale the adjustment by the error
                current_lists[i, j] -= errors[i, i] * (current_lists[i, j] - np.mean(current_lists[i]))
    
    else:
        print(f"Warning: Maximum iterations ({max_iterations}) reached without convergence.")
    print('Adjustment done!')
    return current_lists

def generate_scenarios(data, cor_matrix, z_sgm, n_trails, n_iv, maxit, per_risk, rate):
  print('Starting generating scenarios!')
  # some random sequence
  rsl = list()
  for y in range(n_iv):
    rs = list(range(n_trails))
    random.shuffle(rs)
    rsl.append(rs)

  # parameters list
  sv      = dict()      # key (s)
  ev      = dict()      # key (s, i)
  ev_a    = dict()      # ev adjusted for div
  # other items
  trails  = dict()
  trail_rate = dict()
  iiiv    = dict()
  # main function
  for s in data.index: 
    data_s= data.loc[s]
    # opt parameters - start values
    sv[s] = data_s['i_cost']
    ## stage i ##
    for i in range(n_trails):
      trails[(s,i)] = list()
      trails[(s,i)].append(data_s.loc['i'])
      trail_rate[(s,i)] = list()
      trail_rate[(s,i)].append(data_s.loc['o'])
    ## stage ii ##
    # first half
    vo = halfnorm.rvs(loc=data_s.loc['ii_med'], scale=(data_s.loc['ii_med'] - data_s.loc['ii_low'])/z_sgm, size=int(n_trails/2))
    iiv  = [data_s.loc['ii_med'] * 2 - x for x in vo]
    # second half
    vt = halfnorm.rvs(loc=data_s.loc['ii_med'], scale=(data_s.loc['ii_high'] - data_s.loc['ii_med'])/z_sgm, size=int(n_trails/2))
    iiv = [y for x in [iiv, vt] for y in x]
    random.shuffle(iiv)
    for i in range(n_trails):
      trails[(s,i)].append(iiv[i])
      trail_rate[(s,i)].append(max(pow(pow(1+data_s.loc['o'],2) * iiv[i] / data_s.loc['i'], 1/3)-1,min_g)) # using 3% as the minimum growth rate
    ## stage iii ##
    # first half
    vo = halfnorm.rvs(loc=data_s.loc['iii_med'], scale=(data_s.loc['iii_med'] - data_s.loc['iii_low'])/z_sgm, size=int(n_trails/2))
    iiiv[s]  = [data_s.loc['iii_med'] * 2 - x for x in vo]
    # second half
    vt = halfnorm.rvs(loc=data_s.loc['iii_med'], scale=(data_s.loc['iii_high'] - data_s.loc['iii_med'])/z_sgm, size=int(n_trails/2))
    iiiv[s] = [y for x in [iiiv[s], vt] for y in x]
    random.shuffle(iiiv[s])

  # put intermediate results for correlation adjustment if iteration is > 0.
  lists = list()
  for s in data.index:
    lists.append(iiiv[s])

  # Adjust lists to meet desired inter-list correlations
  if maxit > 0:
    desired_inter_list_corrs = list()
    for s in data.index:
      desired_inter_list_corrs.append(cor_matrix.loc[s].to_list())
    desired_inter_list_corrs = np.array(desired_inter_list_corrs)    
    adjusted_lists = adjust_lists_to_desired_inter_correlations(lists, desired_inter_list_corrs, maxit)
  else:
    adjusted_lists = lists

  # Print inter-list correlations of adjusted lists
  # print("Adjusted Lists:")
  # for i in range(len(data.index)):
  #     for j in range(i + 1, len(data.index)):
  #         corr_between_lists = np.corrcoef(adjusted_lists[i], adjusted_lists[j])[0, 1]
  #         print(f"Correlation between List {i+1} and List {j+1}: {corr_between_lists}")
  # Print final inter-list correlations of adjusted lists
  # print("\nFinal Inter-List Correlations:")
  # print(cor_matrix)
  # adjusted_inter_list_corrs = np.corrcoef(adjusted_lists)
  # print(adjusted_inter_list_corrs)

  s_i = 0
  for s in data.index:
    data_s= data.loc[s]
    ## stage iii continue ##
    iiiv = adjusted_lists[s_i]
    s_i += 1
    for i in range(n_trails):
      trails[(s,i)].append(iiiv[i])
      trail_rate[(s,i)].append(max(pow((1+data_s.loc['o']) * iiiv[i] / data_s.loc['i'], 1/3)-1, min_g))
    ## stage iv ##
    ivv = iiiv/data_s.loc['ii_med'] - data_s.loc['iii_med']/data_s.loc['ii_med']
    for y in range(n_iv):
      ivv = [x * data_s.loc['iv_up']/100 if x > 0 else x * data_s.loc['iv_do']/100 for x in ivv]
      ivv = [ivv[i] for i in rsl[y]]
      for i in range(n_trails):
        val = trails[(s,i)][-1] * min((1 + data_s.loc['iv_rate']/100 + data_s.loc['x_div']/100 + ivv[i]), data_s.loc['iv_cap']/100)
        trails[(s,i)].append(val)
        if y == 0:
          trail_rate[(s,i)].append(max(pow(val/trails[(s,i)][0]*(1+data_s.loc['o']),1/3)-1,min_g))
        else:
          trail_rate[(s,i)].append(max(pow(val/trails[(s,i)][y-1],1/3)-1,min_g))
    ## stage v  & stored parameters for the optimisation ##
    for i in range(n_trails):
      if per_risk == 1:
        cb = min(max(np.random.normal(0, 0.2), -1), 1)
        if cb > 0:
          ar = data_s.loc['x_rate'] + cb * (9 - data_s.loc['x_rate'])
        else:
          ar = data_s.loc['x_rate'] + cb * (data_s.loc['x_rate'] - 0)
        per = (data_s.loc['x_roe'] - ar) / (rate - ar/100) / data_s.loc['x_roe']
      else:
        per = data_s.loc['Per']
      # opt parameters - end values
      ev[(s, i)] = np.round(max(trails[(s,i)][-1], 0) * per, 2)
      # adjusting ev with div but after tax
      ev_a[(s, i)] = np.round(max(trails[(s,i)][-1], 0) * per * pow(1+data_s.loc['x_div']/150, n_iv), 2)
      
      # trails[(s,i)].append(max(trails[(s,i)][-1], 0) * per)
  return sv, ev, ev_a, trails, trail_rate

def simple_calculation(data, n_trails, n_iv, ev, trails, sc, rate, p_tiles):
  check_rate = 0
  val_s   = dict()
  for s in data.index:
    val     = list()
    for i in range(n_trails):
      # do not discount E before maturity
      # val.append(max(npf.npv(rate, trails[(s,i)]).round(1), 0))
      val.append(max(np.round(ev[(s, i)] / pow(1+rate, n_iv+2), 1), 0))
    if check_rate == 1:
      sum = 0
      for i in range(n_trails):
        sum += ev[(s, i)]/n_trails/10
      age = pow(sum/data['iii_med'].loc[s], 1/(n_iv))
      ag  = pow(np.percentile([value for (s, i), value in ev.items() if s == s], p_tiles)/10/data['iii_med'].loc[s], 1/(n_iv))
      print(s)
      print(np.round(ag, 2))
      print(np.round(age, 2))
    val_s[s]   = np.round(np.percentile(val, p_tiles),2)
  return val_s

def no_regret_price(data, n_trails, n_iv, ev, trails, grate, crate, p_tile_low, p_tile_high, trail_rate):
  ud_min    = dict()
  hd_max    = dict()
  cd_min_80 = dict()
  cd_max_80 = dict()
  cd_min_50 = dict()
  cd_max_50 = dict()
  prices    = dict()
  eps       = dict()
  for s in data.index:
    min_list = list()
    max_list = list()
    cmin_list= list()
    cmax_list= list()
    # fogret about the discount rate for now
    for i in range(n_trails):
      i_list = list()
      igd_list = list()
      icd_list = list()
      trail_rate[(s,i)] = [min_g if isinstance(x, float) and np.isnan(x) else x for x in trail_rate[(s,i)]]
      peg = [3*pow(x,2)-7*x+3 for x in trail_rate[(s,i)]]
      for k in range(len(trails[(s,i)])):
        peg_j = max(peg[k], 0.5) * (1 + 1.65 * random.uniform(-peg_std, peg_std)) # try nomral distribution or uniform distribution np.random.normal(0, peg_std)
        pe_j  = max(trail_rate[(s,i)][k] * peg_j * 100, 2)
        prices[(s,i,k)] = max(trails[(s,i)][k] * pe_j, 0)
        eps[(s,i,k)] = trails[(s,i)][k]
        i_list.append(prices[(s,i,k)])
        igd_list.append(prices[(s,i,k)]/pow(1+grate, k+1))
        icd_list.append(prices[(s,i,k)]/pow(1+crate, k+1))
      min_list.append(min(i_list))
      max_list.append(max(igd_list))
      cmin_list.append(min(icd_list))
      cmax_list.append(max(icd_list))
    ud_min[s]     = np.round(np.percentile(min_list, 100-p_tile_low),2)         # no discount
    hd_max[s]     = np.round(np.percentile(max_list, p_tile_low),2)         # high discount
    cd_min_80[s]  = np.round(np.percentile(cmin_list, p_tile_low),2)        # low discount 80% chance
    cd_max_80[s]  = np.round(np.percentile(cmax_list, p_tile_low),2)
    cd_min_50[s]  = np.round(np.percentile(cmin_list, p_tile_high),2)       # low discount 50% chance
    cd_max_50[s]  = np.round(np.percentile(cmax_list, p_tile_high),2)
  return ud_min, hd_max, cd_min_80, cd_max_80, cd_min_50, cd_max_50, prices, eps