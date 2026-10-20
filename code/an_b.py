import numpy as np,pandas as pd,warnings,json;warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from core import df,y,groups,countries,FEATSETS,LAG,EX,NF
yv=y.values;g=groups.values;C=countries.values
F=FEATSETS['v10']; ENV=FEATSETS['env_only']; HIST=FEATSETS['history_only']
rf=lambda:RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=20,class_weight='balanced',random_state=0,n_jobs=2)
xg=lambda:xgb.XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,reg_lambda=20,subsample=.8,colsample_bytree=.8,random_state=0,n_jobs=2)
ml=lambda:make_pipeline(StandardScaler(),MLPClassifier((16,),alpha=.1,max_iter=500,early_stopping=True,random_state=0))
def stack_fit(tr,te,cols):
    a=np.where(tr)[0]; b=np.where(te)[0]; O=np.zeros((len(a),3))
    for ia,ib in StratifiedGroupKFold(4,shuffle=True,random_state=1).split(a,yv[a],g[a]):
        for j,m in enumerate([rf,xg,ml]): O[ib,j]=m().fit(df.loc[a[ia],cols],yv[a[ia]]).predict_proba(df.loc[a[ib],cols])[:,1]
    meta=LogisticRegression(class_weight='balanced',max_iter=1000).fit(O,yv[a])
    T=np.column_stack([m().fit(df.loc[a,cols],yv[a]).predict_proba(df.loc[b,cols])[:,1] for m in [rf,xg,ml]])
    return T,meta.predict_proba(T)[:,1]
tn=sorted(df[df.country=='NIGERIA'].t.unique()); out=[]
for H in (1,3,6):
    ys=[];S={k:[] for k in ['persistence','xgb','rf','stack','stack_history_only','stack_env_only']}
    for T0 in tn[30::(2 if H==1 else H)]:
        tr=(df.t<T0).values; te=((df.t>=T0)&(df.t<T0+H)&(df.country=='NIGERIA')).values
        if te.sum()<10 or len(set(yv[te]))<2: continue
        ys.append(yv[te]); S['persistence'].append(df.log_cases_lag1.values[te])
        Tm,sp=stack_fit(tr,te,F); S['rf'].append(Tm[:,0]); S['xgb'].append(Tm[:,1]); S['stack'].append(sp)
        S['stack_history_only'].append(stack_fit(tr,te,HIST)[1]); S['stack_env_only'].append(stack_fit(tr,te,ENV)[1])
    yy=np.concatenate(ys); rng=np.random.RandomState(2)
    row={'horizon_months':H,'windows':len(ys),'test_rows':len(yy)}
    for k,v in S.items(): row[k]=roc_auc_score(yy,np.concatenate(v))
    for ref in ['persistence','xgb']:
        d=[]
        for b in range(500):
            ix=rng.choice(len(ys),len(ys)); yb=np.concatenate([ys[i] for i in ix])
            if len(set(yb))<2: continue
            d.append(roc_auc_score(yb,np.concatenate([S['stack'][i] for i in ix]))-roc_auc_score(yb,np.concatenate([S[ref][i] for i in ix])))
        row[f'd_stack_{ref}']=np.mean(d); row[f'd_stack_{ref}_lo']=np.percentile(d,2.5); row[f'd_stack_{ref}_hi']=np.percentile(d,97.5)
    out.append(row); print(row,flush=True)
pd.DataFrame(out).round(3).to_csv('res/tables/forward.csv',index=False)
