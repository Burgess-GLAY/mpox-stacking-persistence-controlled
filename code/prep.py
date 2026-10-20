import numpy as np, pandas as pd
def load(path='data/mpox_unit_month_panel.csv'):
    df=pd.read_csv(path)
    df['outbreak']=(df['new_cases_monthly'].fillna(0)>0).astype(int)
    df['temp_mean_c']=df['temp_mean']-273.15; df['precip_total_mm']=df['precip_total']*1000
    for l in [1,2,3]:
        df[f'temp_mean_c_lag{l}']=df[f'temp_mean_lag{l}']-273.15
        df[f'precip_total_mm_lag{l}']=df[f'precip_total_lag{l}']*1000
    df['month_sin']=np.sin(2*np.pi*df['month']/12); df['month_cos']=np.cos(2*np.pi*df['month']/12)
    df=df.sort_values(['country','admin_unit','year','month'])
    df['cum_forest_loss_pct']=df.groupby(['country','admin_unit'])['cum_forest_loss_pct'].ffill()
    df['log_cases']=np.log1p(df['new_cases_monthly'].fillna(0))
    for l in [1,2,3]:
        df[f'log_cases_lag{l}']=df.groupby(['country','admin_unit'])['log_cases'].shift(l).fillna(0)
    df=df.dropna(subset=['temp_mean','soil_moisture','precip_total','cum_forest_loss_pct'])
    F=['population_matched','treecover2000_pct','cum_forest_loss_pct','temp_mean_c','soil_moisture','precip_total_mm',
       'temp_mean_c_lag1','soil_moisture_lag1','precip_total_mm_lag1','temp_mean_c_lag2','soil_moisture_lag2','precip_total_mm_lag2',
       'temp_mean_c_lag3','soil_moisture_lag3','precip_total_mm_lag3','month_sin','month_cos','log_cases_lag1','log_cases_lag2','log_cases_lag3']
    raw=[c for c in F if c not in('month_sin','month_cos')]
    for c in raw:
        m=df.groupby('country')[c].transform('mean'); s=df.groupby('country')[c].transform('std').replace(0,1)
        df[c+'_znorm']=(df[c]-m)/s
    NF=[c+'_znorm' for c in raw]+['month_sin','month_cos']
    return df.reset_index(drop=True),NF
if __name__=='__main__':
    df,NF=load(); print(df.shape, df.outbreak.mean())
    print(df.groupby('country').agg(n=('year','size'),pos=('outbreak','mean'),units=('admin_unit','nunique'),y0=('year','min'),y1=('year','max')))
    print(df.groupby('country').month.nunique()); print(df.spatial_level.value_counts())
    # months per unit/time span
    print(df.groupby(['country']).apply(lambda g:(g.year*12+g.month).agg(['min','max'])))
