import numpy as np,pandas as pd,warnings,json;warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest,mutual_info_classif
import xgboost as xgb
from feat2 import load2
df,NF,EX=load2()
y=df.outbreak; groups=df.country+'|'+df.admin_unit; countries=df.country
LAG=[c for c in NF if 'log_cases' in c]; ENV=[c for c in NF if c not in LAG]
FEATSETS={'v10':NF+EX,'v9':NF,'env_only':ENV,'history_only':LAG+EX}
RF={'max_depth':[2,3,4,5,6],'min_samples_leaf':[10,20,30,50],'n_estimators':[100,150,300]}
XG={'max_depth':[1,2,3,4],'reg_lambda':[5.,10.,20.,50.,100.],'reg_alpha':[0.,1.,5.],'subsample':[.5,.6,.8],'colsample_bytree':[.5,.6,.8],'learning_rate':[.02,.03,.05],'n_estimators':[100,150,300]}
ML={'hidden_layer_sizes':[(8,),(16,),(32,),(32,16)],'alpha':[.01,.05,.1,.3]}
def tune(est,pd_,Xt,yt,gt,n_iter,rs):
    cv=StratifiedGroupKFold(n_splits=min(3,yt.sum(),len(yt)-yt.sum()),shuffle=True,random_state=rs)
    s=RandomizedSearchCV(est,pd_,n_iter=n_iter,cv=cv,scoring='roc_auc',random_state=rs,n_jobs=2); s.fit(Xt,yt,groups=gt); return s.best_estimator_,s.best_params_
def fit_stack(Xtr,ytr,Xte,rs=42,return_models=False):
    gtr=groups.loc[ytr.index]
    if len(Xtr)<500:
        sel=SelectKBest(mutual_info_classif,k=min(8,Xtr.shape[1])).fit(Xtr,ytr); cols=Xtr.columns[sel.get_support()].tolist(); Xtr=Xtr[cols]; Xte=Xte[cols]
    sc=StandardScaler().fit(Xtr); Xs=sc.transform(Xtr); Xtes=sc.transform(Xte)
    npos=ytr.sum(); spw=(len(ytr)-npos)/max(npos,1)
    rf,rfp=tune(RandomForestClassifier(class_weight='balanced',random_state=rs,n_jobs=2),RF,Xtr,ytr,gtr,10,rs)
    xg,xgp=tune(xgb.XGBClassifier(scale_pos_weight=spw,eval_metric='logloss',random_state=rs,n_jobs=2),XG,Xtr,ytr,gtr,12,rs)
    Xsd=pd.DataFrame(Xs,columns=Xtr.columns,index=Xtr.index)
    ml,mlp=tune(MLPClassifier(max_iter=500,random_state=rs,early_stopping=True),ML,Xsd,ytr,gtr,8,rs)
    icv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=rs); oof=np.zeros((len(Xtr),3))
    Xa=Xtr.reset_index(drop=True); ya=ytr.reset_index(drop=True); ga=gtr.reset_index(drop=True); Xsa=pd.DataFrame(Xs,columns=Xtr.columns)
    for a,b in icv.split(Xa,ya,groups=ga):
        oof[b,0]=RandomForestClassifier(**rfp,class_weight='balanced',random_state=rs,n_jobs=2).fit(Xa.iloc[a],ya.iloc[a]).predict_proba(Xa.iloc[b])[:,1]
        oof[b,1]=xgb.XGBClassifier(**xgp,scale_pos_weight=spw,eval_metric='logloss',random_state=rs,n_jobs=2).fit(Xa.iloc[a],ya.iloc[a]).predict_proba(Xa.iloc[b])[:,1]
        oof[b,2]=MLPClassifier(**mlp,max_iter=500,random_state=rs,early_stopping=True).fit(Xsa.iloc[a],ya.iloc[a]).predict_proba(Xsa.iloc[b])[:,1]
    meta=LogisticRegression(class_weight='balanced',max_iter=1000).fit(oof,ya)
    tm=np.column_stack([rf.predict_proba(Xte)[:,1],xg.predict_proba(Xte)[:,1],ml.predict_proba(Xtes)[:,1]])
    info={'rf':{k:(v if not isinstance(v,tuple) else list(v)) for k,v in rfp.items()},'xgb':xgp,'mlp':{k:(list(v) if isinstance(v,tuple) else v) for k,v in mlp.items()},'meta_coef':meta.coef_[0].tolist(),'meta_intercept':float(meta.intercept_[0])}
    if return_models: return tm,meta.predict_proba(tm)[:,1],info,dict(rf=rf,xgb=xg,mlp=ml,meta=meta,scaler=sc,cols=list(Xtr.columns))
    return tm,meta.predict_proba(tm)[:,1],info
