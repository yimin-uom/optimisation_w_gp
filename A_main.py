import pandas as pd
import B_pre_processing, B_generate_scenarios
import C_optimisation_sizing as C_optimisation
import D_optimisation_timing as D_optimisation

# read csv inputs
data_path   = 'C:\\Users\\yimzhang3\\OneDrive - The University of Melbourne\\Documents\\Local Files\\Optimisation_w_gp\\optimisation_w_gp\\inputs\\i_scenarios.csv'
cor_path    = 'C:\\Users\\yimzhang3\\OneDrive - The University of Melbourne\\Documents\\Local Files\\Optimisation_w_gp\\optimisation_w_gp\\inputs\\cor_matrix.csv'
out_path    = 'C:\\Users\\yimzhang3\\OneDrive - The University of Melbourne\\Documents\\Local Files\\Optimisation_w_gp\\optimisation_w_gp\\outputs\\'

# control parameters
pre         = 0
gs          = 0
sc          = 0
copt        = 1
dopt        = 0

# fixed parameters
# generate scenarios
tiles       = [2.5, 50, 97.5]       # percentile of low, med, high
z_sgm       = 1.65                  # 1.65 sigma is 90% of normal distribution
n_trails    = 10000
n_iv        = 10
maxit       = 0                     # adjust for the correlation, I think it is fine, the correlation is not accurate anyway.
# simple calculation
grate       = 0.2
crate       = 0.1
p_tile_low  = 20
p_tile_high = 50

def second_max_min(df, cols):
    df['Second Max'] = df.apply(lambda x: x.nlargest(2).iloc[1] if x.nlargest(2).iloc[0] != x.nlargest(2).iloc[1] else 0, axis=1)
    df['Max']        = df.apply(lambda x: x.max(), axis=1)
    df['Cap']        = df.apply(lambda x: (x['Max'] + x['Second Max']) / 2, axis=1)
    df['Second Min'] = df.apply(lambda x: x.nsmallest(2).iloc[1] if x.nsmallest(2).iloc[0] != x.nsmallest(2).iloc[1] else 0, axis=1)
    df['Min']        = df.apply(lambda x: x.min(), axis=1)
    df['Floor']      = df.apply(lambda x: (x['Min'] + x['Second Min']) / 2, axis=1)
    df['Conditional Floor'] = df.apply(lambda x: x['Floor'] if x['Floor'] <= 0.8 * x['Cap'] else 0.8 * x['Cap'], axis=1)
    return df

if pre == 1:
    B_pre_processing.calc_correlation(data_path, cor_path, out_path, crate)

data        = pd.read_csv(data_path, header=0).set_index('index')
cor_matrix  = pd.read_csv(cor_path, header=0).set_index('index')

if gs == 1:
    per_risk = 1
    sv, ev, ev_a, trails, trail_rate = B_generate_scenarios.generate_scenarios(data, cor_matrix, z_sgm, n_trails, n_iv, maxit, per_risk, crate)
    df = pd.DataFrame(list(ev.items()), columns=['Name', 'Value'])
    df[['Code', 'Scenario']] = pd.DataFrame(df['Name'].tolist(), index=df.index)
    df = df.drop(columns=['Name'])
    df = pd.pivot_table(df, values='Value', index=['Code'], columns=['Scenario'], aggfunc="sum")
    df.to_csv(out_path+'Ending_value.csv', index=True)
    df = pd.DataFrame(list(ev_a.items()), columns=['Name', 'Value'])
    df[['Code', 'Scenario']] = pd.DataFrame(df['Name'].tolist(), index=df.index)
    df = df.drop(columns=['Name'])
    df = pd.pivot_table(df, values='Value', index=['Code'], columns=['Scenario'], aggfunc="sum")
    df.to_csv(out_path+'Ending_value_adjusted.csv', index=True)

if sc == 1:
    val_s = B_generate_scenarios.simple_calculation(data, n_trails, n_iv, ev, trails, sc, crate, p_tile_high)
    df = pd.DataFrame(list(val_s.items()), columns=['Code', 'Central']).set_index('Code')
    # df.to_csv(out_path+'Calc_value_s.csv', index=True)
    ud_min, hd_max, cd_min_80, cd_max_80, cd_min_50, cd_max_50, prices, eps = B_generate_scenarios.no_regret_price(data, n_trails, n_iv, ev, trails, grate, crate, p_tile_low, p_tile_high, trail_rate)
    df_max = pd.DataFrame(list(ud_min.items()), columns=['Code', 'ud_min']).set_index('Code')
    df_min = pd.DataFrame(list(hd_max.items()), columns=['Code', 'hd_max']).set_index('Code')
    df = pd.concat([df, df_max, df_min], axis=1)
    df_max = pd.DataFrame(list(cd_min_80.items()), columns=['Code', 'cd_min_80']).set_index('Code')
    df_min = pd.DataFrame(list(cd_max_80.items()), columns=['Code', 'cd_max_80']).set_index('Code')
    df = pd.concat([df, df_max, df_min], axis=1)
    df_max = pd.DataFrame(list(cd_min_50.items()), columns=['Code', 'cd_min_50']).set_index('Code')
    df_min = pd.DataFrame(list(cd_max_50.items()), columns=['Code', 'cd_max_50']).set_index('Code')
    df_combined = pd.concat([df, df_max, df_min], axis=1)
    df_combined.to_csv(out_path+'Calc_value_prob.csv', index=True)
    df = pd.DataFrame(list(prices.items()), columns=['key', 'value'])
    df[['s', 'i', 'k']] = pd.DataFrame(df['key'].tolist(), index=df.index)
    df = df.drop(columns=['key'])
    df = df[['s', 'i', 'k', 'value']]
    df.to_csv(out_path+'prices.csv', index=False)
    df = pd.DataFrame(list(eps.items()), columns=['key', 'value'])
    df[['s', 'i', 'k']] = pd.DataFrame(df['key'].tolist(), index=df.index)
    df = df.drop(columns=['key'])
    df = df[['s', 'i', 'k', 'value']]
    df.to_csv(out_path+'eps.csv', index=False)
    ctrl = pd.DataFrame(columns=['Code', 'i', 'ii', 'iii', 'iv', 'v'])
    ctrl['Code']    = df_combined.index
    ctrl            = ctrl.set_index('Code')
    ctrl['i']       = df_combined['Central']
    ctrl['ii']      = df_combined['ud_min']
    ctrl['iii']     = df_combined['hd_max']
    ctrl['iv']      = (df_combined['cd_min_80']+df_combined['cd_max_80'])/2
    ctrl['v']       = (df_combined['cd_min_50']*2+df_combined['cd_max_50'])/3
    ctrl = second_max_min(ctrl, ['i', 'ii', 'iii', 'iv', 'v'])
    ctrl.to_csv(out_path+'i_mannual_control.csv', index=True)

if copt == 1:
    first_stage_scenarios   = 10000
    run_stage_2 = 0
    second_stage_scenarios  = 10000
    df = pd.read_csv(out_path+'Ending_value_adjusted.csv').set_index('Code')
    prices  = pd.read_csv(out_path+'prices.csv').set_index(['s', 'i', 'k'])
    mannual = pd.read_csv(out_path+'/i_mannual_control.csv').set_index('Code')
    initial = pd.read_csv(out_path+'/i_mannual_initial.csv').set_index('index')
    obj_1 = C_optimisation.opt_single_stage(data, first_stage_scenarios, n_iv, grate, crate, df, prices, mannual, initial)
    obj_2 = dict()
    if run_stage_2 == 0:
        exit()
    # for i in range(second_stage_scenarios):
    #     obj_2[i] = C_optimisation.opt_second_stage(data, first_stage_scenarios, n_iv, grate, crate, df, prices, mannual, initial, i)
    print(obj_1)
    print(obj_2)
    relax_1 = 1.1
    relax_2 = 1.01
    C_optimisation.opt_multi_stage(data, first_stage_scenarios, n_iv, grate, crate, df, prices, mannual, initial, obj_1, obj_2, relax_1, relax_2, range(second_stage_scenarios))

if dopt == 1:
    df = pd.read_csv(out_path+'Ending_value_adjusted.csv').set_index('Code')
    prices  = pd.read_csv(out_path+'prices.csv').set_index(['s', 'i', 'k'])
    for s in df.index:
        print(s)
        D_optimisation.rules(data, n_trails, n_iv, grate, crate, df, prices, mannual, initial)
        exit()