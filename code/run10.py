import sys,time,pickle
from core import *
mode,fs,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
X=df[FEATSETS[fs]].reset_index(drop=True); yy=y.reset_index(drop=True); t0=time.time(); infos=[]
if mode=='primary':
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed); P=np.zeros((len(X),5))
    for i,(a,b) in enumerate(cv.split(X,yy,groups=groups)):
        tm,sp,info=fit_stack(X.iloc[a],yy.iloc[a],X.iloc[b]); P[b,:3]=tm;P[b,3]=sp;P[b,4]=i; infos.append(info)
        print(fs,seed,'fold',i,round(roc_auc_score(yy.iloc[b],sp),3),round(time.time()-t0),flush=True)
    np.save(f'res/primary_{fs}_{seed}.npy',P); json.dump(infos,open(f'res/primary_{fs}_{seed}_info.json','w'))
else:
    L=np.zeros((len(X),4))
    for c in ['DRC','NIGERIA','UGANDA']:
        a=(countries!=c).values;b=(countries==c).values
        tm,sp,info=fit_stack(X[a],yy[a],X[b]); L[b,:3]=tm;L[b,3]=sp; infos.append(info)
        print(fs,'LOCO',c,round(roc_auc_score(yy[b],sp),3),round(time.time()-t0),flush=True)
    np.save(f'res/loco_{fs}_{seed}.npy',L); json.dump(infos,open(f'res/loco_{fs}_{seed}_info.json','w'))
