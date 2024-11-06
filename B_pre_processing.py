def calc_correlation(data_path, cor_path, out_path, crate):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    # read csv inputs
    data = pd.read_csv(data_path, header=0).set_index('index')

    # time traces
    demand = dict()
    spd = pd.DataFrame(columns = ['index', 'EPS', 'Book', 'ROA', 'ROA_BIT', 'Lg', 'mp'])
    cor_matrix = list()
    cor_pair   = dict()
    inrows = ['Operating Revenue', 'Gross Profit', 'EBITDA', 'EBIT', 'Pretax Income', 'Net Income']
    barows = ['Common Stock Equity', 'Total Liabilities Net Minority Interest', 'Long Term Debt', 'Ordinary Shares Number']

    for s in data.index:
        sif = yf.Ticker(s)
        sif.info
        hist = sif.history(period='1y')
        demand[s]   = np.array(hist['Open'].to_list())
        df = sif.income_stmt
        # print(df.index)
        df_i = df[df.index.isin(inrows)]
        df_i = df_i.reindex(inrows)
        df_i.to_csv(out_path + s + '_income.csv')
        df = sif.balance_sheet
        # print(df.index)
        df_b = df[df.index.isin(barows)]
        df_b = df_b.reindex(barows)
        df_b.to_csv(out_path + s + '_balance.csv')
        eps = df_i.iloc[5, 0]/df_b.iloc[3, 0]
        if eps> 0.8 * data.loc[s, 'i'] and eps < 1.2 * data.loc[s, 'i']:
            bok = (df_b.iloc[0, 0]+df_b.iloc[1, 0])/df_b.iloc[3, 0]
            roa = df_i.iloc[5, 0]/(df_b.iloc[0, 0]+df_b.iloc[1, 0])
            bit = df_i.iloc[3, 0]/(df_b.iloc[0, 0]+df_b.iloc[1, 0])
            lg  = (df_b.iloc[0, 0]+df_b.iloc[1, 0]) / df_b.iloc[0, 0]
            mp  = roa * lg * (bit - crate) / ((data.loc[s, 'iii_med'] - data.loc[s, 'iii_low']) / bok) / (1+data.loc[s, 'iv_up']/100) / 2 * 10
            spd = spd._append({ 'index': s, 'EPS': eps, 'Book':bok,
                                'ROA': roa, 'ROA_BIT': bit,
                                'Lg': lg, 'mp': mp}, ignore_index = True)
        else:
            print(s + ' data is likely wrong! Do it mannually or remove it!')
    spd.to_csv(out_path + 'stats.csv', index=False)
    for s in data.index:
        cor_l = list()
        for s2 in data.index:
            if s2 == s:
                cor_l.append(1)
            else:
                try:
                    cor_pair[(s, s2)] = cor_pair[(s2, s)]
                except:
                    cor_pair[(s, s2)] = np.corrcoef(demand[s], demand[s2])[0, 1]
                cor_l.append(cor_pair[(s, s2)])
        cor_matrix.append(cor_l)
    df = pd.DataFrame(cor_matrix, index=data.index, columns=data.index)
    df.to_csv(cor_path)
