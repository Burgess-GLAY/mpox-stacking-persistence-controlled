import numpy as np,pandas as pd,pickle,json,shap
from core import *
X=df[FEATSETS['v10']].reset_index(drop=True); yy=y.reset_index(drop=True)
tm,sp,info,mod=fit_stack(X,yy,X,return_models=True)
json.dump(info,open('res/final_fit_info.json','w'),indent=1); pickle.dump({k:v for k,v in mod.items()},open('res/final_models.pkl','wb'))
rf,xg,ml,meta,sc=mod['rf'],mod['xgb'],mod['mlp'],mod['meta'],mod['scaler']
Xs=sc.transform(X)
r=shap.TreeExplainer(rf).shap_values(X); r=r[:,:,1] if np.ndim(r)==3 else (r[1] if isinstance(r,list) else r)
x=shap.TreeExplainer(xg).shap_values(X)
bg=shap.sample(Xs,100,random_state=42); idx=np.random.RandomState(42).choice(len(Xs),200,replace=False)
m=shap.KernelExplainer(lambda z:ml.predict_proba(z)[:,1],bg).shap_values(Xs[idx],nsamples=200)
imp=pd.DataFrame({'feature':X.columns,'rf':np.abs(r).mean(0),'xgb':np.abs(x).mean(0),'mlp':np.abs(m).mean(0)})
imp['combined']=(meta.coef_[0][0]*np.abs(r).mean(0)+meta.coef_[0][1]*np.abs(x).mean(0))
imp=imp.sort_values('xgb',ascending=False); imp.to_csv('res/tables/shap_level1.csv',index=False); print(imp.round(4).to_string(index=False))
np.save('res/shap_rf.npy',r); np.save('res/shap_xgb.npy',x); np.save('res/shap_mlp.npy',m); np.save('res/shap_mlp_idx.npy',idx)
# Level 2: exact Shapley for linear meta on probabilities: phi_j = coef_j * (p_j - mean p_j)
Pm=np.column_stack([rf.predict_proba(X)[:,1],xg.predict_proba(X)[:,1],ml.predict_proba(Xs)[:,1]])
phi=meta.coef_[0]*(Pm-Pm.mean(0)); share=np.abs(phi).mean(0)/np.abs(phi).mean(0).sum()
L2=pd.DataFrame({'learner':['RF','XGB','MLP'],'coef':meta.coef_[0],'mean_abs_phi':np.abs(phi).mean(0),'share':share}); print(L2); L2.to_csv('res/tables/shap_level2.csv',index=False)
# by country
rows=[]
for c in ['DRC','NIGERIA','UGANDA']:
    mm=(countries.values==c); s_=np.abs(phi[mm]).mean(0); rows.append({'country':c,**dict(zip(['RF','XGB','MLP'],s_/s_.sum()))})
print(pd.DataFrame(rows).round(3)); pd.DataFrame(rows).to_csv('res/tables/shap_level2_country.csv',index=False)
# group-level: history vs environment share of combined attribution
hist=[c for c in X.columns if c in FEATSETS['history_only']]; 
tot=imp.set_index('feature').xgb; print('XGB SHAP share history features:',tot[hist].sum()/tot.sum())
