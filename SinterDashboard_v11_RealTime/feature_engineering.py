import pandas as pd, numpy as np
from date_utils import sort_by_date

LEAKAGE_SQ  = ['%+40MM','%40+25MM','%25+10MM','%10+5MM','%5 MM','%MPS']
LEAKAGE_PP  = ['%+40MM','%-40+25MM','%-25+10MM','%-10+5MM','%-5MM','%MPS',
               '%6.3MM(TI)','%-0.5MM','RDI','RI','G.I.','B.I.','Fines Generation']
TARGETS     = ['TI','RDI','RI']

def engineer(df, is_pp=False):
    df = df.copy()

    # Rolling/lag features below are computed on ROW ORDER, not on the date
    # value itself — if the sheet isn't stored in chronological order (the
    # source workbooks for this dataset aren't), '_r3/_r7/_r14' rolling
    # averages and '_lag1/_lag3/_lag7' features end up averaging/shifting
    # across the wrong days entirely. Guarantee chronological order here so
    # every caller (training, live ingestion, ad-hoc scripts) gets correct
    # features regardless of how the source file happened to be sorted.
    date_col = 'Date' if is_pp else 'DATE'
    if date_col in df.columns:
        df = sort_by_date(df, date_col)

    get = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    cao,sio2,mgo,al,feo,tfe = get('%CaO'),get('%SiO2'),get('%MgO'),get('%Al2O3'),get('%FeO'),get('%T(Fe)')
    b2_col = 'Basicity(B2)' if is_pp else 'Basicity  (B2)'
    b2 = get(b2_col) if b2_col in df.columns else cao/sio2.replace(0,np.nan)

    if '%CaO' in df and '%SiO2' in df:
        df['B2_calc']     = cao / sio2.replace(0,np.nan)
        df['B3']          = (cao+mgo) / sio2.replace(0,np.nan)
        df['B4']          = (cao+mgo) / (sio2+al).replace(0,np.nan)
    if '%MgO' in df and '%Al2O3' in df:
        df['MgO_Al2O3_r'] = mgo / al.replace(0,np.nan)
        df['Gangue_load'] = al + sio2
        df['Al2O3_SiO2']  = al / sio2.replace(0,np.nan)
    if '%FeO' in df and '%T(Fe)' in df:
        df['FeO_TFe']     = feo / tfe.replace(0,np.nan)
    if '%FeO' in df:
        df['B2_x_FeO']    = b2 * feo
    if '%Al2O3' in df:
        df['Al2O3_x_B2']  = al * b2
    if '%MgO' in df and '%Al2O3' in df:
        df['MgO_x_Al2O3'] = mgo * al

    if is_pp:
        if 'Carbon Rate' in df and 'Production' in df:
            df['Fuel_intensity'] = df['Carbon Rate'] / df['Production'].replace(0,np.nan)
        if 'BTP Temperature' in df and 'Machine speed' in df:
            df['BTP_per_speed']  = df['BTP Temperature'] / df['Machine speed'].replace(0,np.nan)
        if 'Furnace Temp.' in df and 'Feed' in df:
            df['Heat_input']     = df['Furnace Temp.'] * df['Feed'] / 1e4
        if 'ESP Inlet Temp.' in df and 'BTP Temperature' in df:
            df['ESP_BTP_ratio']  = df['ESP Inlet Temp.'] / df['BTP Temperature'].replace(0,np.nan)
        if 'WB 1 Pressure' in df and 'Machine speed' in df:
            df['Suction_speed']  = df['WB 1 Pressure'] * df['Machine speed']

    roll_src = [c for c in ['%FeO','%CaO','%MgO','%SiO2','%Al2O3',
                             'Basicity(B2)','Basicity  (B2)','MgO/Al2O3',
                             'Carbon Rate','Machine speed','BTP Temperature']
                if c in df.columns]
    for col in roll_src:
        s = col.replace(' ','_').replace('%','pct').replace('(','').replace(')','').replace('/','_').replace('.','')
        df[f'{s}_r3']  = df[col].rolling(3,  min_periods=1).mean()
        df[f'{s}_r7']  = df[col].rolling(7,  min_periods=1).mean()
        df[f'{s}_r14'] = df[col].rolling(14, min_periods=1).mean()

    lag_src = [c for c in ['%FeO','%CaO','%MgO','%Al2O3','Basicity(B2)','Basicity  (B2)','MgO/Al2O3']
               if c in df.columns]
    for col in lag_src:
        s = col.replace(' ','_').replace('%','pct').replace('(','').replace(')','').replace('/','_').replace('.','')
        for lag in [1,3,7]: df[f'{s}_lag{lag}'] = df[col].shift(lag)

    return df
