import sys; exec(open('an_b.py').read().split("tn=sorted")[0])
import json,time
RAWMAP={c:(c[:-6] if c.endswith('_znorm') else c) for c in F}
def Xnorm(train_mask,cols):
    X=df[[RAWMAP[c] for c in cols]].copy(); X.columns=cols
    for c in cols:
        if c.endswith('_znorm'):
            for k in np.unique(C):
                m=(C==k); tm=m&train_mask
                mu=df.loc[tm,RAWMAP[c]].mean(); sd=df.loc[tm,RAWMAP[c]].std() or 1.0
                X.loc[m,c]=(df.loc[m,RAWMAP[c]]-mu)/(sd if sd>0 else 1.0)
    return X
def stack_fit_X(Xall,tr,te):
    a=np.where(tr)[0]; b=np.where(te)[0]; O=np.zeros((len(a),3))
    for ia,ib in StratifiedGroupKFold(4,shuffle=True,random_state=1).split(a,yv[a],g[a]):
        for j,m in enumerate([rf,xg,ml]): O[ib,j]=m().fit(Xall.iloc[a[ia]],yv[a[ia]]).predict_proba(Xall.iloc[a[ib]])[:,1]
    meta=LogisticRegression(class_weight='balanced',max_iter=1000).fit(O,yv[a])
    T=np.column_stack([m().fit(Xall.iloc[a],yv[a]).predict_proba(Xall.iloc[b])[:,1] for m in [rf,xg,ml]])
    return T,meta.predict_proba(T)[:,1]
res={}
t0=time.time()
# E3 paired stack vs average / xgb under fixed-hyperparameter protocol (exp2 arrays)
names=['RF_all','XGB_all','MLP_all','LR_hist','ET_all','XGB_env','XGB_hist','LR_all']
U=pd.unique(g); idx={u:np.where(g==u)[0] for u in U}
rng=np.random.RandomState(0); BS=[np.concatenate([idx[u] for u in rng.choice(U,len(U))]) for _ in range(1000)]
D={'stack_minus_avg':[], 'stack_minus_xgb':[], 'stack_minus_rf':[]}
for s in range(3):
    st=np.load(f'exp2_S_stack_orig3_{s}.npy'); av=np.load(f'exp2_S_avg_orig3_{s}.npy'); P=np.load(f'exp2_P_{s}.npy')
    d={k:[] for k in D}
    for ix in BS:
        if len(set(yv[ix]))<2: continue
        a=roc_auc_score(yv[ix],st[ix]); d['stack_minus_avg'].append(a-roc_auc_score(yv[ix],av[ix])); d['stack_minus_xgb'].append(a-roc_auc_score(yv[ix],P[ix,1])); d['stack_minus_rf'].append(a-roc_auc_score(yv[ix],P[ix,0]))
    for k in D: D[k].append(np.mean(d[k]))
res['E3']={k:[float(np.mean(v))] for k,v in D.items()}
# per-resample stats across all seeds combined
allc={k:[] for k in D}
for s in range(3):
    st=np.load(f'exp2_S_stack_orig3_{s}.npy'); av=np.load(f'exp2_S_avg_orig3_{s}.npy'); P=np.load(f'exp2_P_{s}.npy')
    for k,ref in [('stack_minus_avg',av),('stack_minus_xgb',P[:,1]),('stack_minus_rf',P[:,0])]:
        allc[k].append([roc_auc_score(yv[ix],st[ix])-roc_auc_score(yv[ix],ref[ix]) for ix in BS[:500] if len(set(yv[ix]))>1])
res['E3']={k:{'mean':float(np.mean(np.mean(v,axis=0))),'lo':float(np.percentile(np.mean(v,axis=0),2.5)),'hi':float(np.percentile(np.mean(v,axis=0),97.5))} for k,v in allc.items()}
print('E3',res['E3'],round(time.time()-t0),flush=True); json.dump(res,open('res/exp5.json','w'),indent=1)
# E2 primary: full-panel vs train-only standardization, fixed hyperparameters, 3 seeds
rows=[]
for seed in range(3):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed); PF=np.zeros((len(yv),4)); PT=np.zeros((len(yv),4))
    for a,b in cv.split(df,yv,g):
        tr=np.zeros(len(yv),bool); tr[a]=True; te=np.zeros(len(yv),bool); te[b]=True
        Tm,sp=stack_fit(tr,te,F); PF[b,:3]=Tm; PF[b,3]=sp
        Xn=Xnorm(tr,F); Tm,sp=stack_fit_X(Xn,tr,te); PT[b,:3]=Tm; PT[b,3]=sp
    r={'seed':seed}
    for nm,P in [('full',PF),('trainonly',PT)]:
        for j,k in enumerate(['rf','xgb','mlp','stack']): r[f'{nm}_{k}']=roc_auc_score(yv,P[:,j])
    rows.append(r); print('E2',r,round(time.time()-t0),flush=True)
pd.DataFrame(rows).round(4).to_csv('res/tables/std_primary.csv',index=False)
# E2/E4 forward: full-panel vs train-only std; and v9 vs v10 (features); stack, 1/3/6 months
tn=sorted(df[df.country=='NIGERIA'].t.unique()); V9=FEATSETS['v9']; out=[]
for H in (1,3,6):
    ys=[];S={k:[] for k in ['persistence','stack_full','stack_trainonly','stack_v9','xgb_trainonly','rf_trainonly']}
    for T0 in tn[30::(2 if H==1 else H)]:
        tr=(df.t<T0).values; te=((df.t>=T0)&(df.t<T0+H)&(df.country=='NIGERIA')).values
        if te.sum()<10 or len(set(yv[te]))<2: continue
        ys.append(yv[te]); S['persistence'].append(df.log_cases_lag1.values[te])
        S['stack_full'].append(stack_fit(tr,te,F)[1]); S['stack_v9'].append(stack_fit(tr,te,V9)[1])
        Xn=Xnorm(tr,F); Tm,sp=stack_fit_X(Xn,tr,te); S['stack_trainonly'].append(sp); S['rf_trainonly'].append(Tm[:,0]); S['xgb_trainonly'].append(Tm[:,1])
    yy=np.concatenate(ys); row={'horizon':H,'windows':len(ys),'rows':len(yy)}
    for k,v in S.items(): row[k]=roc_auc_score(yy,np.concatenate(v))
    r2=np.random.RandomState(3)
    for ref in ['persistence','xgb_trainonly','stack_v9']:
        d=[]
        for _ in range(500):
            ix=r2.choice(len(ys),len(ys)); yb=np.concatenate([ys[i] for i in ix])
            if len(set(yb))<2: continue
            d.append(roc_auc_score(yb,np.concatenate([S['stack_trainonly' if ref!='stack_v9' else 'stack_full'][i] for i in ix]))-roc_auc_score(yb,np.concatenate([S[ref][i] for i in ix])))
        row[f'd_{ref}']=np.mean(d); row[f'd_{ref}_lo']=np.percentile(d,2.5); row[f'd_{ref}_hi']=np.percentile(d,97.5)
    out.append(row); print('E4',row,round(time.time()-t0),flush=True)
pd.DataFrame(out).round(3).to_csv('res/tables/forward_robust.csv',index=False)
# E1 LOCO + forward-1m with higher-capacity RF/XGB vs A1-like config
cfg={'rf_A1':lambda:RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=10,class_weight='balanced',random_state=0,n_jobs=2),
'rf_big':lambda:RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=1,class_weight='balanced',random_state=0,n_jobs=2),
'xgb_A1':lambda:xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,reg_lambda=5,reg_alpha=5,subsample=.8,colsample_bytree=.6,random_state=0,n_jobs=2),
'xgb_big':lambda:xgb.XGBClassifier(n_estimators=600,max_depth=6,learning_rate=.05,reg_lambda=0,subsample=.8,colsample_bytree=.8,random_state=0,n_jobs=2)}
rowsE=[]
for k,mk in cfg.items():
    r={'config':k}
    for c in ['DRC','NIGERIA','UGANDA']:
        tr=(C!=c); te=(C==c); r['loco_'+c]=roc_auc_score(yv[te],mk().fit(df.loc[tr,F],yv[tr]).predict_proba(df.loc[te,F])[:,1])
    for H in (1,3,6):
        ys=[];ps=[]
        for T0 in tn[30::(2 if H==1 else H)]:
            tr=(df.t<T0).values; te=((df.t>=T0)&(df.t<T0+H)&(df.country=='NIGERIA')).values
            if te.sum()<10 or len(set(yv[te]))<2: continue
            ys.append(yv[te]); ps.append(mk().fit(df.loc[tr,F],yv[tr]).predict_proba(df.loc[te,F])[:,1])
        r[f'fwd{H}']=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
    rowsE.append(r); print('E1',r,round(time.time()-t0),flush=True)
pd.DataFrame(rowsE).round(3).to_csv('res/tables/capacity_transfer.csv',index=False)
print('DONE')
