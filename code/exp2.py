import numpy as np,pandas as pd,warnings,sys;warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from feat2 import load2
df,NF,EX=load2(); y=df.outbreak.values; C=df.country.values; g=(df.country+'|'+df.admin_unit).values
LAG=[c for c in NF if 'log_cases' in c]; ENV=[c for c in NF if c not in LAG]
HIST=LAG+EX; ALL=NF+EX
def rf(): return RandomForestClassifier(n_estimators=250,max_depth=5,min_samples_leaf=20,class_weight='balanced',random_state=0,n_jobs=2)
def xg(): return xgb.XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,reg_lambda=20,subsample=.8,colsample_bytree=.8,random_state=0,n_jobs=2)
def lr(): return make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=3000,C=0.3))
def mlp(): return make_pipeline(StandardScaler(),MLPClassifier((16,),alpha=.1,max_iter=500,early_stopping=True,random_state=0))
def et(): return ExtraTreesClassifier(n_estimators=250,max_depth=6,min_samples_leaf=15,class_weight='balanced',random_state=0,n_jobs=2)
BASE={'RF_all':(rf,ALL),'XGB_all':(xg,ALL),'MLP_all':(mlp,ALL),'LR_hist':(lr,HIST),'ET_all':(et,ALL),'XGB_env':(xg,ENV),'XGB_hist':(xg,HIST),'LR_all':(lr,ALL)}
def lg(p): p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
def run(seed):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed); names=list(BASE); P=np.zeros((len(y),len(names))); fold=np.zeros(len(y),int)
    Pm={}  # stack variants
    S={k:np.zeros(len(y)) for k in ['stack_orig3','stack_div5','stack_all8','avg_orig3','avg_all8']}
    for i,(a,b) in enumerate(cv.split(df,y,g)):
        fold[b]=i
        # inner OOF
        icv=StratifiedGroupKFold(4,shuffle=True,random_state=seed+100); O=np.zeros((len(a),len(names)))
        for ia,ib in icv.split(a,y[a],g[a]):
            for j,n in enumerate(names):
                m,cols=BASE[n]; O[ib,j]=m().fit(df.loc[a[ia],cols],y[a[ia]]).predict_proba(df.loc[a[ib],cols])[:,1]
        T=np.zeros((len(b),len(names)))
        for j,n in enumerate(names):
            m,cols=BASE[n]; T[:,j]=m().fit(df.loc[a,cols],y[a]).predict_proba(df.loc[b,cols])[:,1]
        P[b]=T
        def meta(idx):
            mm=LogisticRegression(class_weight='balanced',max_iter=2000,C=1.0).fit(lg(O[:,idx]),y[a]); return mm.predict_proba(lg(T[:,idx]))[:,1]
        ix=lambda ns:[names.index(n) for n in ns]
        S['stack_orig3'][b]=meta(ix(['RF_all','XGB_all','MLP_all']))
        S['stack_div5'][b]=meta(ix(['XGB_all','LR_hist','XGB_env','MLP_all','ET_all']))
        S['stack_all8'][b]=meta(list(range(len(names))))
        S['avg_orig3'][b]=T[:,ix(['RF_all','XGB_all','MLP_all'])].mean(1)
        S['avg_all8'][b]=T.mean(1)
    return names,P,S,fold
res=[]
for seed in range(3):
    names,P,S,fold=run(seed); np.save(f'exp2_P_{seed}.npy',P); np.save(f'exp2_fold_{seed}.npy',fold)
    for k,v in S.items(): np.save(f'exp2_S_{k}_{seed}.npy',v)
    r={n:roc_auc_score(y,P[:,j]) for j,n in enumerate(names)}; r.update({k:roc_auc_score(y,v) for k,v in S.items()}); res.append(r); print(seed,{k:round(v,3) for k,v in r.items()},flush=True)
R=pd.DataFrame(res); print(R.mean().round(3).to_string()); print('std',R.std().round(3).max())
