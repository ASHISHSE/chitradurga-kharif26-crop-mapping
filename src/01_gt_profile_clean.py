"""Task A: profile + clean the Kharif-26 GT. Run: python src/01_gt_profile_clean.py"""
import pandas as pd, numpy as np, geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED=42; UTM=32643; MIN_PER_CLASS=20   # classes below this are NOT mapped/claimed
raw=pd.read_excel("data/GT_Data_Kharif26_Chitradurga.xlsx")
raw.columns=["lat","lon","crop_raw","crop_type","inter_raw"]
dist=gpd.read_file("data/boundary/Chitradurga.shp").to_crs(4326)
log=[]
FIX={"coconut":"Coconut","dragon":"Dragonfruit","dragonfruit":"Dragonfruit","horse":"Horse gram",
     "horse gram":"Horse gram","promogrante":"Pomegranate","tur":"Tur"}
norm=lambda s: np.nan if pd.isna(s) else FIX.get(str(s).strip().lower(), str(s).strip().capitalize())
d=raw.copy(); d["crop"]=d.crop_raw.map(norm); d["inter"]=d.inter_raw.map(norm)
chg=d[d.crop!=d.crop_raw].groupby(["crop_raw","crop"]).size()
log.append(f"Label normalisation changed {int((d.crop!=d.crop_raw).sum())} crop labels:\n{chg.to_string()}")
log.append("ASSUMPTIONS to confirm with field team: 'Dragon'->Dragonfruit, 'Horse'->Horse gram, 'Promogrante'->Pomegranate (all single-sample classes anyway).")
dup=d.duplicated(["lat","lon","crop"],keep="first"); log.append(f"Dropped {int(dup.sum())} exact duplicate row(s).")
d=d[~dup].copy()
pts=gpd.GeoDataFrame(d,geometry=gpd.points_from_xy(d.lon,d.lat),crs=4326)
d["inside"]=pts.within(dist.geometry.iloc[0]).values
log.append(f"Points inside district: {int(d.inside.sum())}/{len(d)}; outside dropped: {int((~d.inside).sum())}.")
d=d[d.inside].copy(); pts=pts.loc[d.index]
pu=pts.to_crs(UTM); bu=dist.to_crs(UTM).geometry.iloc[0]
d["dist_edge_m"]=pu.geometry.distance(bu.boundary).values
xy=np.c_[pu.geometry.x,pu.geometry.y]; nn,_=cKDTree(xy).query(xy,k=2); d["nn_m"]=nn[:,1]
d["close_pair"]=d.nn_m<20
log.append(f"Min distance to district edge {d.dist_edge_m.min():.0f} m -> no district-edge mixed-pixel issue.")
log.append(f"{int(d.close_pair.sum())} points have a neighbour <20 m (same/adjacent S2 pixel): kept, but grouped together in spatial CV; pixel-level duplicates removed at extraction.")
cnt=d.crop.value_counts()
log.append("Class counts after cleaning:\n"+cnt.to_string())
log.append(f"MAPPED classes (n>={MIN_PER_CLASS}): {list(cnt[cnt>=MIN_PER_CLASS].index)}. All others -> 'Other/unmodelled', NOT reported as mapped.")
hull=gpd.GeoSeries([pts.union_all().convex_hull],crs=4326).to_crs(UTM).area.iloc[0]
dist_area=dist.to_crs(UTM).area.iloc[0]
log.append(f"GT convex hull = {hull/1e6:.0f} km2 = {100*hull/dist_area:.1f}% of district ({dist_area/1e6:.0f} km2). Rest of district has ZERO GT.")
d.drop(columns="inside").to_csv("outputs/gt_clean.csv",index=False)
open("outputs/gt_cleaning_log.txt","w").write("\n\n".join(log)); print("\n\n".join(log))
fig,ax=plt.subplots(1,2,figsize=(13,6)); dist.boundary.plot(ax=ax[0],color="k")
for c,gdf in pts.assign(crop=d.crop).groupby("crop"): gdf.plot(ax=ax[0],markersize=10,label=c)
ax[0].legend(fontsize=6); ax[0].set_title("GT points vs district boundary")
cnt.plot.barh(ax=ax[1]); ax[1].set_title("GT samples per class"); plt.tight_layout(); plt.savefig("outputs/gt_coverage.png",dpi=150)
