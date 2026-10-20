import numpy as np,warnings;warnings.filterwarnings('ignore')
exec(open('exp2.py').read().split("res=[]")[0])
from sklearn.metrics import roc_auc_score
names=list(BASE); ci=[names.index(n) for n in ['RF_all','XGB_all','MLP_all','LR_hist','ET_all','XGB_env']]
oh=lambda idx: np.column_stack([(C[idx]==c).astype(float) for c in ['DRC','NIGERIA','UGANDA']])
out=[]
for seed in range(3):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed); G={k:np.zeros(len(y)) for k in ['plain6','gated','gated_hist']}
    for a,b in cv.split(df,y,g):
        icv=StratifiedGroupKFold(4,shuffle=True,random_state=seed+100); O=np.zeros((len(a),len(ci)))
        for ia,ib in icv.split(a,y[a],g[a]):
            for j,k in enumerate(ci):
                m,cols=BASE[names[k]]; O[ib,j]=m().fit(df.loc[a[ia],cols],y[a[ia]]).predict_proba(df.loc[a[ib],cols])[:,1]
        T=np.column_stack([BASE[names[k]][0]().fit(df.loc[a,BASE[names[k]][1]],y[a]).predict_proba(df.loc[b,BASE[names[k]][1]])[:,1] for k in ci])
        La,Lb=lg(O),lg(T)
        G['plain6'][b]=LogisticRegression(class_weight='balanced',max_iter=2000).fit(La,y[a]).predict_proba(Lb)[:,1]
        Za=np.column_stack([La]+[La*oh(a)[:,[i]] for i in range(3)]+[oh(a)]); Zb=np.column_stack([Lb]+[Lb*oh(b)[:,[i]] for i in range(3)]+[oh(b)])
        G['gated'][b]=LogisticRegression(class_weight='balanced',max_iter=3000,C=.3).fit(Za,y[a]).predict_proba(Zb)[:,1]
        # gate on history regime instead of country
        h=lambda idx:((df.log_cases_lag1.values[idx]>0)|(df.log_cases_lag2.values[idx]>0)|(df.log_cases_lag3.values[idx]>0)).astype(float)[:,None]
        Ha=np.column_stack([La,La*h(a),h(a)]); Hb=np.column_stack([Lb,Lb*h(b),h(b)])
        G['gated_hist'][b]=LogisticRegression(class_weight='balanced',max_iter=3000,C=.3).fit(Ha,y[a]).predict_proba(Hb)[:,1]
    r={k:(roc_auc_score(y,v),*[roc_auc_score(y[C==c],v[C==c]) for c in ['DRC','NIGERIA','UGANDA']]) for k,v in G.items()}
    out.append(r); print(seed,{k:np.round(v,3).tolist() for k,v in r.items()},flush=True)
