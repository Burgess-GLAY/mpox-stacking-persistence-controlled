import numpy as np,pandas as pd,json,matplotlib,shap
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
from core import df,y,groups,countries,FEATSETS
CR='#B22222'; BK='#000000'; GR='#6B6B6B'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5})
OUT='figures/'
def save(fig,n): fig.savefig(OUT+n+'.pdf',bbox_inches='tight'); plt.close(fig)
yv=y.values;C=countries.values;g=groups.values;U=pd.unique(g);idx={u:np.where(g==u)[0] for u in U}
P=np.load('res/primary_v10_42.npy'); rc=np.load('res/recal_seed42.npy'); persist=df.log_cases_lag1.values
def lab(a,t): a.text(-0.16,1.04,t,transform=a.transAxes,fontsize=9,fontweight='bold')
# Fig2 ROC + PR
fig,(a,b)=plt.subplots(1,2,figsize=(6.6,3.1))
mods=[('Persistence',persist,':',1.0,GR),('MLP',P[:,2],'-.',1.1,BK),('Random Forest',P[:,0],'--',1.1,BK),('XGBoost',P[:,1],(0,(1,1)),1.4,BK),('Stacked ensemble',P[:,3],'-',2.0,CR)]
for n,s,ls,lw,c in mods:
    f,t,_=roc_curve(yv,s); a.plot(f,t,ls=ls,lw=lw,color=c,label=f'{n} ({roc_auc_score(yv,s):.3f})')
    p,r,_=precision_recall_curve(yv,s); b.plot(r,p,ls=ls,lw=lw,color=c,label=f'{n} ({average_precision_score(yv,s):.3f})')
a.plot([0,1],[0,1],color='#999',lw=.6); b.axhline(yv.mean(),color='#999',lw=.6)
a.set_xlabel('False-positive rate'); a.set_ylabel('True-positive rate'); b.set_xlabel('Recall'); b.set_ylabel('Precision')
a.legend(loc='lower right',title='AUC-ROC',title_fontsize=7.5); b.legend(loc='lower left',title='Average precision',title_fontsize=7.5); b.set_ylim(0,1.02)
lab(a,'(a)'); lab(b,'(b)'); fig.tight_layout(); save(fig,'Fig2')
# Fig3 ablation + regimes
rb=pd.read_csv('res/tables/regime_boot.csv',index_col=0)
def boot(score,B=1000):
    r=np.random.RandomState(0); v=[]
    for _ in range(B):
        ix=np.concatenate([idx[u] for u in r.choice(U,len(U))])
        if len(set(yv[ix]))>1: v.append(roc_auc_score(yv[ix],score[ix]))
    return np.percentile(v,[2.5,97.5])
sets=[('All 25 features','v10',CR,'o'),('Original 20 features','v9',BK,'s'),('History features only (8)','history_only',BK,'D'),('Environment, demography, calendar (17)','env_only',GR,'^')]
fig,(a,b)=plt.subplots(1,2,figsize=(6.8,2.8),gridspec_kw={'width_ratios':[1,1.15]})
for i,(n,k,c,m) in enumerate(sets):
    s=np.load(f'res/primary_{k}_42.npy')[:,3]; au=roc_auc_score(yv,s); lo,hi=boot(s)
    a.errorbar(au,3-i,xerr=[[au-lo],[hi-au]],fmt=m,color=c,ms=5,capsize=2.5,lw=1)
    a.text(hi+.012,3-i,f'{au:.3f}',va='center',fontsize=7.5)
a.set_yticks(range(4)); a.set_yticklabels([s[0] for s in sets][::-1]); a.set_xlim(.5,1.04); a.set_xlabel('Pooled AUC-ROC (95% interval)'); a.axvline(.5,color='#999',lw=.6)
rg=pd.read_csv('res/tables/regime.csv').set_index('features')
for j,(reg,off) in enumerate([('onset',-.15),('continuation',.15)]):
    pass
ypos={'v10':3,'v9':2,'history_only':1,'env_only':0}
for k,(n,key,c,m) in zip(['v10','v9','history_only','env_only'],sets):
    for reg,off,fc in [('onset',-.14,'white'),('continuation',.14,c)]:
        v=rg.loc[k,f'{reg}_auc']; b.plot(v,ypos[k]+off,marker=m,ms=5.5,color=c,mfc=fc,mec=c,ls='none')
b.set_yticks([]); b.set_xlim(.45,1.0); b.set_xlabel('AUC-ROC within regime'); b.axvline(.5,color='#999',lw=.6)
b.plot([],[],'o',mfc='white',mec=BK,color=BK,label='New onset'); b.plot([],[],'o',color=BK,label='Continuation'); b.legend(loc='lower right')
lab(a,'(a)'); b.text(-0.05,1.04,'(b)',transform=b.transAxes,fontsize=9,fontweight='bold'); fig.tight_layout(); save(fig,'Fig3')
# Fig4 forward
fw=pd.read_csv('res/tables/forward.csv'); x=np.arange(3)
fig,(a,b)=plt.subplots(1,2,figsize=(6.6,2.9))
for col,l,ls,mk,c,lw in [('persistence','Persistence',':','o',GR,1.1),('xgb','XGBoost','--','s',BK,1.1),('rf','Random Forest','-.','^',BK,1.1),('stack','Stacked ensemble','-','D',CR,1.9)]:
    a.plot(x,fw[col],ls=ls,marker=mk,ms=5,lw=lw,color=c,label=l)
a.set_xticks(x); a.set_xticklabels(['1 month','3 months','6 months']); a.set_ylabel('Pooled AUC-ROC (Nigeria)'); a.set_xlabel('Forecast horizon'); a.set_ylim(.7,.92); a.legend(loc='upper right')
for k,(col,l,mk,c,off) in enumerate([('persistence','Stack − persistence','o',CR,-.08),('xgb','Stack − XGBoost','s',BK,.08)]):
    d=fw[f'd_stack_{col}']; lo=fw[f'd_stack_{col}_lo']; hi=fw[f'd_stack_{col}_hi']
    b.errorbar(x+off,d,yerr=[d-lo,hi-d],fmt=mk,color=c,ms=5,capsize=2.5,lw=1,label=l)
b.axhline(0,color='#666',lw=.7); b.set_xticks(x); b.set_xticklabels(['1 month','3 months','6 months']); b.set_xlabel('Forecast horizon'); b.set_ylabel('Paired AUC difference'); b.legend(loc='upper right')
lab(a,'(a)'); lab(b,'(b)'); fig.tight_layout(); save(fig,'Fig4')
# Fig5 evaluation tiers
S={'DRC':(0.915,0.013),'NIGERIA':(0.946,0.002),'UGANDA':(0.636,0.113)}; LO={'DRC':.576,'NIGERIA':.547,'UGANDA':.632}
fig,ax=plt.subplots(figsize=(4.2,2.8)); w=.36
ax.bar(np.arange(3)-w/2,[S[k][0] for k in S],w,yerr=[S[k][1] for k in S],color='white',edgecolor=BK,hatch='////',capsize=3,label='Unit-grouped cross-validation',error_kw=dict(ecolor='#444',lw=1))
ax.bar(np.arange(3)+w/2,[LO[k] for k in S],w,color=CR,label='Leave-one-country-out')
for i,k in enumerate(S): ax.text(i-w/2,S[k][0]+S[k][1]+.015,f'{S[k][0]:.3f}',ha='center',fontsize=7.5); ax.text(i+w/2,LO[k]+.015,f'{LO[k]:.3f}',ha='center',fontsize=7.5)
ax.axhline(.5,ls=':',color=GR,lw=.9); ax.set_xticks(range(3)); ax.set_xticklabels(['DRC','Nigeria','Uganda']); ax.set_ylabel('AUC-ROC (stacked ensemble)'); ax.set_ylim(0,1.3); ax.set_yticks(np.arange(0,1.01,.2)); ax.legend(loc='upper center',ncol=1,bbox_to_anchor=(.5,1.04)); save(fig,'Fig5')
# Fig6 calibration
fig,ax=plt.subplots(figsize=(3.3,3.3)); ax.plot([0,1],[0,1],':',color=GR,lw=.9,label='Perfect calibration')
for s,l,col,ls,mk in [(P[:,3],'Raw output',BK,'--','s'),(rc,'After recalibration',CR,'-','o')]:
    q=np.quantile(s,np.linspace(0,1,11)); xs=[];ys=[]
    for lo,hi in zip(q[:-1],q[1:]):
        m=(s>=lo)&(s<=hi); xs.append(s[m].mean()); ys.append(yv[m].mean())
    ax.plot(xs,ys,ls=ls,marker=mk,ms=4,lw=1.4,color=col,label=l)
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed outbreak frequency'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.legend(loc='upper left'); save(fig,'Fig6')
# Fig7 SHAP + level 2
cm=LinearSegmentedColormap.from_list('bk_cr',[BK,CR])
X=df[FEATSETS['v10']].reset_index(drop=True); sx=np.load('res/shap_xgb.npy')
pretty=lambda s:s.replace('_znorm','').replace('log_cases_lag','Log cases, lag ').replace('nat_prev','Country burden, prior month').replace('since_last','Months since last case').replace('roll3','Mean log cases, 3 mo').replace('hist_rate','Share of months with cases').replace('trend','Case trend').replace('month_cos','Month (cos)').replace('month_sin','Month (sin)').replace('cum_forest_loss_pct','Cumulative forest loss').replace('precip_total_mm_lag','Precipitation, lag ').replace('precip_total_mm','Precipitation').replace('soil_moisture_lag','Soil moisture, lag ').replace('soil_moisture','Soil moisture').replace('temp_mean_c_lag','Temperature, lag ').replace('temp_mean_c','Temperature').replace('treecover2000_pct','Tree cover (2000)').replace('population_matched','Population')
Xn=X.copy(); Xn.columns=[pretty(c) for c in X.columns]

sxm=np.abs(sx).mean(0); order=np.argsort(sxm)[::-1][:10]
fig,(ax,bx)=plt.subplots(1,2,figsize=(6.8,3.2),gridspec_kw={'width_ratios':[2.3,1]})
rng=np.random.RandomState(1)
for r,j in enumerate(order):
    v=X.iloc[:,j].values; lo,hi=np.percentile(v,[5,95]); cval=np.clip((v-lo)/(hi-lo+1e-9),0,1)
    ax.scatter(sx[:,j],9-r+rng.normal(0,.11,len(v)),c=cval,cmap=cm,s=3.5,linewidths=0,rasterized=True)
ax.set_yticks(range(10)); ax.set_yticklabels([Xn.columns[j] for j in order][::-1]); ax.axvline(0,color='#999',lw=.6)
ax.set_xlabel('SHAP value (contribution to log-odds)'); ax.set_ylim(-.6,9.6)
sm=plt.cm.ScalarMappable(cmap=cm); cb=fig.colorbar(sm,ax=ax,pad=.01,fraction=.04,ticks=[0,1]); cb.ax.set_yticklabels(['Low','High']); cb.set_label('Feature value',fontsize=7.5); cb.outline.set_visible(False)
l2=pd.read_csv('res/tables/shap_level2.csv'); nm={'RF':'RF','XGB':'XGB','MLP':'MLP'}; o=l2.set_index('learner').loc[['XGB','RF','MLP']]
bx.bar([nm[i] for i in o.index],o.share*100,color=[CR,BK,'white'],edgecolor=BK,hatch=None)
bx.patches[2].set_hatch('////')
for i,v in enumerate(o.share*100): bx.text(i,v+1.5,f'{v:.1f}%',ha='center',fontsize=7.5)
bx.set_ylabel('Share of meta-learner attribution (%)'); bx.set_ylim(0,85)
lab(ax,'(a)'); bx.text(-0.32,1.04,'(b)',transform=bx.transAxes,fontsize=9,fontweight='bold')
fig.tight_layout(); save(fig,'Fig7')
print('ok')
