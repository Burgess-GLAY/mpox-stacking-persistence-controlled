import numpy as np,pandas as pd
from prep import load
def load2():
    df,NF=load()
    df['t']=df.year*12+df.month
    df=df.sort_values(['country','admin_unit','t']).reset_index(drop=True)
    k=['country','admin_unit']
    obs=df.groupby(k).outbreak
    prev=obs.shift(1)  # y_{t-1}
    df['prev_out']=prev.fillna(0)
    # months since last outbreak month (before t), capped 12; unseen -> 12
    def since(g):
        out=[];last=None
        for i,v in enumerate(g.values):
            out.append(12 if last is None else min(i-last,12))
            if v==1: last=i
        return pd.Series(out,index=g.index)
    df['since_last']=df.groupby(k).outbreak.apply(since).reset_index(level=[0,1],drop=True)
    df['since_last']=df['since_last'].astype(float)
    df['hist_rate']=(obs.cumsum()-df.outbreak)/ (df.groupby(k).cumcount().replace(0,np.nan))
    df['hist_rate']=df['hist_rate'].fillna(0)
    df['roll3']=df[['log_cases_lag1','log_cases_lag2','log_cases_lag3']].mean(axis=1)
    df['trend']=df.log_cases_lag1-df.log_cases_lag2
    # national/country burden at t-1: share of units in same country with cases at t-1 (uses only past info)
    tmp=df.groupby(['country','t']).prev_out.mean().rename('nat_prev').reset_index()
    df=df.merge(tmp,on=['country','t'],how='left')
    for c in ['since_last','hist_rate','roll3','trend','nat_prev']:
        m=df.groupby('country')[c].transform('mean'); s=df.groupby('country')[c].transform('std').replace(0,1)
        df[c+'_znorm']=(df[c]-m)/s
    EXTRA=[c+'_znorm' for c in ['since_last','hist_rate','roll3','trend','nat_prev']]
    return df,NF,EXTRA
