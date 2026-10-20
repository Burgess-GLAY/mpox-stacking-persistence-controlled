import numpy as np,pandas as pd,json,glob
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from core import df,y,groups,countries,FEATSETS,LAG,EX,NF
yv=y.values; C=countries.values; g=groups.values; U=pd.unique(g); idx={u:np.where(g==u)[0] for u in U}
M=['rf','xgb','mlp','stack']; CS=['DRC','NIGERIA','UGANDA']
T={}
# 1 repeats
rows=[]
for f in sorted(glob.glob('res/primary_v10_*[0-9].npy')):
    s=int(f.split('_')[-1][:-4]); P=np.load(f)
    for j,m in enumerate(M):
        r={'seed':s,'model':m,'pooled_auc':roc_auc_score(yv,P[:,j]),'pooled_ap':average_precision_score(yv,P[:,j])}
        for c in CS: r[c]=roc_auc_score(yv[C==c],P[C==c,j])
        fa=[roc_auc_score(yv[P[:,4]==i],P[P[:,4]==i,j]) for i in range(5)]; r['fold_mean']=np.mean(fa); r['fold_sd']=np.std(fa,ddof=1)
        rows.append(r)
R=pd.DataFrame(rows); R.to_csv('res/tables/primary_by_seed.csv',index=False)
S=R.groupby('model').agg(['mean','std']).round(3); S.to_csv('res/tables/primary_summary.csv'); print(S[[c for c in S.columns if c[0] in('pooled_auc','pooled_ap','DRC','NIGERIA','UGANDA','fold_mean')]])
# 2 bootstrap on seed 42
P=np.load('res/primary_v10_42.npy'); rng=np.random.RandomState(0)
def boot(fn,B=1000):
    out=[]
    for b in range(B):
        ix=np.concatenate([idx[u] for u in rng.choice(U,len(U))])
        if len(set(yv[ix]))<2: continue
        out.append(fn(ix))
    return pd.DataFrame(out)
def fn(ix):
    r={m:roc_auc_score(yv[ix],P[ix,j]) for j,m in enumerate(M)}
    r['d_stack_rf']=r['stack']-r['rf']; r['d_stack_xgb']=r['stack']-r['xgb']; r['d_stack_mlp']=r['stack']-r['mlp']
    r['persist']=roc_auc_score(yv[ix],df.log_cases_lag1.values[ix]); r['d_stack_persist']=r['stack']-r['persist']
    for c in CS:
        m=C[ix]==c
        if len(set(yv[ix][m]))>1: r[c]=roc_auc_score(yv[ix][m],P[ix,3][m])
    return r
B=boot(fn); ci=lambda s:(round(s.mean(),3),round(s.quantile(.025),3),round(s.quantile(.975),3))
BT=pd.DataFrame({k:ci(B[k].dropna()) for k in B.columns},index=['mean','lo','hi']).T; BT.to_csv('res/tables/bootstrap_seed42.csv'); print(BT)
# 3 baselines on seed-42 folds
fold=P[:,4].astype(int); base={}
base['persistence_lag1']=df.log_cases_lag1.values
base['persistence_prevout']=df.prev_out.values
def cvlr(cols):
    s=np.zeros(len(yv))
    for i in range(5):
        tr=fold!=i; m=LogisticRegression(class_weight='balanced',max_iter=3000,C=.3).fit(df.loc[tr,cols],yv[tr]); s[~tr]=m.predict_proba(df.loc[~tr,cols])[:,1]
    return s
base['logreg_v9_features']=cvlr(NF); base['logreg_v10_features']=cvlr(NF+EX)
rows=[]
for k,s in base.items():
    r={'baseline':k,'pooled':roc_auc_score(yv,s)}; r.update({c:roc_auc_score(yv[C==c],s[C==c]) for c in CS}); rows.append(r)
r={'baseline':'stack_v10','pooled':roc_auc_score(yv,P[:,3])}; r.update({c:roc_auc_score(yv[C==c],P[C==c,3]) for c in CS}); rows.append(r)
BL=pd.DataFrame(rows).round(3); BL.to_csv('res/tables/baselines.csv',index=False); print(BL)
# 4 ablations (seed 42)
rows=[]; Pa={}
for f in ['v10','v9','history_only','env_only']:
    Pa[f]=np.load(f'res/primary_{f}_42.npy')
    for m in ['xgb','stack']:
        j=M.index(m); s=Pa[f][:,j]; r={'features':f,'model':m,'pooled':roc_auc_score(yv,s)}; r.update({c:roc_auc_score(yv[C==c],s[C==c]) for c in CS}); rows.append(r)
AB=pd.DataFrame(rows).round(3); AB.to_csv('res/tables/ablation.csv',index=False); print(AB)
def fd(ix): return {'v10_minus_v9':roc_auc_score(yv[ix],Pa['v10'][ix,3])-roc_auc_score(yv[ix],Pa['v9'][ix,3]),'v10_minus_hist':roc_auc_score(yv[ix],Pa['v10'][ix,3])-roc_auc_score(yv[ix],Pa['history_only'][ix,3]),'v10_minus_env':roc_auc_score(yv[ix],Pa['v10'][ix,3])-roc_auc_score(yv[ix],Pa['env_only'][ix,3])}
BD=boot(fd); BDT=pd.DataFrame({k:ci(BD[k]) for k in BD.columns},index=['mean','lo','hi']).T; BDT.to_csv('res/tables/ablation_boot.csv'); print(BDT)
# 5 LOCO
rows=[]
for f in ['v10','v9']:
    L=np.load(f'res/loco_{f}_42.npy')
    for j,m in enumerate(M):
        r={'features':f,'model':m}; r.update({c:roc_auc_score(yv[C==c],L[C==c,j]) for c in CS}); rows.append(r)
LO=pd.DataFrame(rows).round(3); LO.to_csv('res/tables/loco.csv',index=False); print(LO)
# 6 calibration
s=P[:,3]; lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); rc=np.zeros(len(yv))
for i in range(5):
    tr=fold!=i; m=LogisticRegression().fit(lg(s[tr]).reshape(-1,1),yv[tr]); rc[~tr]=m.predict_proba(lg(s[~tr]).reshape(-1,1))[:,1]
def ece(p,yy,k=10):
    q=np.quantile(p,np.linspace(0,1,k+1)); e=0
    for lo,hi in zip(q[:-1],q[1:]):
        m=(p>=lo)&(p<=hi); e+=m.mean()*abs(p[m].mean()-yy[m].mean())
    return e
cal={'mean_pred_raw':s.mean(),'mean_pred_recal':rc.mean(),'obs_rate':yv.mean(),'brier_raw':brier_score_loss(yv,s),'brier_recal':brier_score_loss(yv,rc),'brier_prevalence':brier_score_loss(yv,np.full(len(yv),yv.mean())),'ece_raw':ece(s,yv),'ece_recal':ece(rc,yv)}
print(cal); json.dump(cal,open('res/tables/calibration.json','w'))
np.save('res/recal_seed42.npy',rc)
fpr,tpr,thr=roc_curve(yv,s); jj=np.argmax(tpr-fpr); print('Youden raw thr',thr[jj],'sens',tpr[jj],'spec',1-fpr[jj])
def conf(p,t):
    pr=p>=t; tp=(pr&(yv==1)).sum(); fp=(pr&(yv==0)).sum(); fn_=((~pr)&(yv==1)).sum(); tn=((~pr)&(yv==0)).sum()
    return dict(thr=round(float(t),3),sens=tp/(tp+fn_),spec=tn/(tn+fp),ppv=tp/max(tp+fp,1),npv=tn/(tn+fn_),acc=(tp+tn)/len(yv))
fpr2,tpr2,thr2=roc_curve(yv,rc); j2=np.argmax(tpr2-fpr2)
th=pd.DataFrame([{'prob':'raw','rule':'0.5',**conf(s,.5)},{'prob':'raw','rule':'Youden',**conf(s,thr[jj])},{'prob':'recalibrated','rule':'0.5',**conf(rc,.5)},{'prob':'recalibrated','rule':'Youden',**conf(rc,thr2[j2])}]).round(3); th.to_csv('res/tables/thresholds.csv',index=False); print(th)
# 7 regime
h0=((df.log_cases_lag1==0)&(df.log_cases_lag2==0)&(df.log_cases_lag3==0)).values
rows=[]
for f in ['v10','v9','env_only','history_only']:
    sc=Pa[f][:,3]; rows.append({'features':f,'onset_auc':roc_auc_score(yv[h0],sc[h0]),'continuation_auc':roc_auc_score(yv[~h0],sc[~h0]),'onset_n':int(h0.sum()),'onset_pos_rate':yv[h0].mean(),'cont_n':int((~h0).sum()),'cont_pos_rate':yv[~h0].mean()})
rows.append({'features':'persistence','onset_auc':np.nan,'continuation_auc':roc_auc_score(yv[~h0],df.log_cases_lag1.values[~h0])})
RG=pd.DataFrame(rows).round(3); RG.to_csv('res/tables/regime.csv',index=False); print(RG)
def fr(ix):
    m1=h0[ix]; r={}
    for f in ['v10','env_only','history_only']:
        if len(set(yv[ix][m1]))>1: r['onset_'+f]=roc_auc_score(yv[ix][m1],Pa[f][ix,3][m1])
    return r
BR=boot(fr,500); print(pd.DataFrame({k:ci(BR[k].dropna()) for k in BR.columns},index=['mean','lo','hi']).T)
pd.DataFrame({k:ci(BR[k].dropna()) for k in BR.columns},index=['mean','lo','hi']).T.to_csv('res/tables/regime_boot.csv')
