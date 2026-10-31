"""Task C: RF vs XGBoost vs (balanced) RF with spatial block CV. Usage: python src/03_train_validate.py"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (classification_report,confusion_matrix,cohen_kappa_score,accuracy_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import config as C
df=pd.read_parquet("outputs/gt_features.parquet")
df=df[df.crop.isin(C.CLASSES)].reset_index(drop=True)      # scope: only adequately sampled classes
feat=[l.strip() for l in open("outputs/feature_names.txt")]
X=df[feat].fillna(df[feat].median()).values; le=LabelEncoder(); y=le.fit_transform(df.crop)
# spatial blocks: 3 km grid cells -> whole blocks held out together (reduces autocorrelation leakage)
BLOCK=3000; xy=df[["pix_x","pix_y"]].values*C.RES
groups=pd.factorize(pd.Series(list(zip((xy[:,0]//BLOCK).astype(int),(xy[:,1]//BLOCK).astype(int)))))[0]
print("blocks:",len(set(groups)))
models={
 "RF_balanced":lambda: RandomForestClassifier(500,class_weight="balanced_subsample",min_samples_leaf=2,n_jobs=-1,random_state=C.SEED),
 "XGB":lambda: XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,
                             random_state=C.SEED,n_jobs=-1,eval_metric="mlogloss"),
}
res={}
nsplit=min(5,len(set(groups)))
for name,mk in models.items():
    pred=np.zeros_like(y)
    for tr,te in GroupKFold(nsplit).split(X,y,groups):
        w=None
        m=mk()
        if name=="XGB":
            cw=len(y[tr])/(len(set(y[tr]))*np.bincount(y[tr],minlength=len(le.classes_)).clip(1)); w=cw[y[tr]]
            m.fit(X[tr],y[tr],sample_weight=w)
        else: m.fit(X[tr],y[tr])
        pred[te]=m.predict(X[te])
    rep=classification_report(y,pred,target_names=le.classes_,output_dict=True,zero_division=0)
    res[name]=dict(OA=accuracy_score(y,pred),kappa=cohen_kappa_score(y,pred),macroF1=rep["macro avg"]["f1-score"],
                   per_class={c:rep[c] for c in le.classes_},
                   confusion=confusion_matrix(y,pred).tolist(),labels=list(le.classes_))
    print(name,{k:round(v,3) for k,v in res[name].items() if isinstance(v,float)})
json.dump(res,open("outputs/metrics_spatialCV.json","w"),indent=2)
best=max(res,key=lambda k:res[k]["macroF1"]); print("best by macro-F1:",best)
final=models[best](); 
if best=="XGB": 
    cw=len(y)/(len(set(y))*np.bincount(y)); final.fit(X,y,sample_weight=cw[y])
else: final.fit(X,y)
joblib.dump(dict(model=final,le=le,feat=feat,medians=df[feat].median().to_dict()),"outputs/model.joblib")
imp=pd.Series(final.feature_importances_,index=feat).sort_values(ascending=False)
imp.to_csv("outputs/feature_importance.csv"); print(imp.head(15))
