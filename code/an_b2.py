exec(open('an_b.py').read().split("tn=sorted")[0])
tn=sorted(df[df.country=='NIGERIA'].t.unique()); out=[]
ys=[];S={k:[] for k in ['persistence','xgb','rf','stack']}; wins=[]
for T0 in tn[len(tn)//2::6]:
    H=6; tr=(df.t<T0).values; te=((df.t>=T0)&(df.t<T0+H)&(df.country=='NIGERIA')).values
    if te.sum()<10 or len(set(yv[te]))<2: continue
    ys.append(yv[te]); wins.append(int(T0)); S['persistence'].append(df.log_cases_lag1.values[te])
    Tm,sp=stack_fit(tr,te,F); S['rf'].append(Tm[:,0]); S['xgb'].append(Tm[:,1]); S['stack'].append(sp)
yy=np.concatenate(ys); row={'scheme':'late_start_6m','windows':len(ys),'test_rows':len(yy)}
for k,v in S.items(): row[k]=roc_auc_score(yy,np.concatenate(v))
print(row); pd.DataFrame([row]).round(3).to_csv('res/tables/forward_alt.csv',index=False)
