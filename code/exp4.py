import sys,json,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from core import *
from sklearn.preprocessing import StandardScaler
X=df[FEATSETS['v10']].reset_index(drop=True); yy=y.reset_index(drop=True); g=groups.reset_index(drop=True); C=countries.reset_index(drop=True)
infos=json.load(open('res/primary_v10_42_info.json'))
cv=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,yy,groups=g))
def mk(info):
    rf=RandomForestClassifier(**info['rf'],class_weight='balanced',random_state=42,n_jobs=-1)
    return rf
def fit_eval(Xa,ya,Xb,yb,rfp,xgp,mlp,spw):
    out={}
    rf=RandomForestClassifier(**rfp,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xa,ya)
    xg=xgb.XGBClassifier(**xgp,scale_pos_weight=spw,eval_metric='logloss',random_state=42,n_jobs=-1).fit(Xa,ya)
    sc=StandardScaler().fit(Xa); hl=tuple(mlp['hidden_layer_sizes']) if isinstance(mlp['hidden_layer_sizes'],list) else mlp['hidden_layer_sizes']
    ml=MLPClassifier(hidden_layer_sizes=hl,alpha=mlp['alpha'],max_iter=500,random_state=42,early_stopping=True).fit(sc.transform(Xa),ya)
    for n,m,tr,te in [('rf',rf,Xa,Xb),('xgb',xg,Xa,Xb),('mlp',ml,sc.transform(Xa),sc.transform(Xb))]:
        out[n]=(roc_auc_score(ya,m.predict_proba(tr)[:,1]),roc_auc_score(yb,m.predict_proba(te)[:,1]) if len(set(yb))>1 else np.nan)
    return out
# A: per-fold train vs test with each fold's selected hyperparameters
rowsA=[]
for i,(a,b) in enumerate(cv):
    npos=yy.iloc[a].sum(); spw=(len(a)-npos)/npos; inf=infos[i]
    o=fit_eval(X.iloc[a],yy.iloc[a],X.iloc[b],yy.iloc[b],inf['rf'],inf['xgb'],inf['mlp'],spw)
    for n,(tr,te) in o.items(): rowsA.append(dict(fold=i,model=n,train=tr,test=te,gap=tr-te))
A=pd.DataFrame(rowsA); A.to_csv('res/tables/overfit_perfold.csv',index=False); print(A.groupby('model')[['train','test','gap']].mean().round(3)); print(A.pivot(index='fold',columns='model',values='gap').round(3))
# B: permutation of labels (within country), final A1 settings
fi=json.load(open('res/final_fit_info.json')); rowsB=[]
for rep in range(3):
    rs=np.random.RandomState(100+rep); yp=yy.copy()
    for c in C.unique():
        m=(C==c).values; yp[m]=rs.permutation(yy[m].values)
    for i,(a,b) in enumerate(cv):
        npos=yp.iloc[a].sum(); spw=(len(a)-npos)/npos
        o=fit_eval(X.iloc[a],yp.iloc[a],X.iloc[b],yp.iloc[b],fi['rf'],fi['xgb'],fi['mlp'],spw)
        for n,(tr,te) in o.items(): rowsB.append(dict(rep=rep,fold=i,model=n,train=tr,test=te))
B=pd.DataFrame(rowsB); B.to_csv('res/tables/overfit_permuted.csv',index=False); print(B.groupby('model')[['train','test']].mean().round(3))
# pooled permuted test AUC? per-fold mean ok
# C: capacity ladder, real labels, per-fold test AUC mean & train AUC mean
rowsC=[]
def ladder(name,mkm,Xa_fn=lambda a:a):
    tr=[];te=[]
    for i,(a,b) in enumerate(cv):
        m=mkm(a).fit(X.iloc[a],yy.iloc[a]); tr.append(roc_auc_score(yy.iloc[a],m.predict_proba(X.iloc[a])[:,1])); te.append(roc_auc_score(yy.iloc[b],m.predict_proba(X.iloc[b])[:,1]))
    rowsC.append(dict(config=name,train=np.mean(tr),test=np.mean(te),gap=np.mean(tr)-np.mean(te)))
for d in [2,4,6,10,None]:
    for l in [1,5,10,30]:
        ladder(f'RF depth={d} leaf={l}',lambda a,d=d,l=l:RandomForestClassifier(n_estimators=100,max_depth=d,min_samples_leaf=l,class_weight='balanced',random_state=42,n_jobs=-1))
for d,lam,n in [(1,5,300),(2,5,300),(3,5,300),(3,1,300),(3,0,300),(5,1,300),(6,1,300),(6,0,600),(8,0,600)]:
    ladder(f'XGB depth={d} lambda={lam} trees={n}',lambda a,d=d,lam=lam,n=n:xgb.XGBClassifier(max_depth=d,reg_lambda=lam,reg_alpha=5 if lam>=5 else 0,n_estimators=n,learning_rate=.05,subsample=.8,colsample_bytree=.6,scale_pos_weight=(len(a)-yy.iloc[a].sum())/yy.iloc[a].sum(),eval_metric='logloss',random_state=42,n_jobs=-1))
Cdf=pd.DataFrame(rowsC); Cdf.to_csv('res/tables/overfit_ladder.csv',index=False); print(Cdf.round(3).to_string())
