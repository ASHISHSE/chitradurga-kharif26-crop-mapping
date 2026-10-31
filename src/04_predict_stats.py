"""Task D: tile-wise prediction, confidence raster, area stats, GT-support mask. Needs outputs/feature_stack.zarr"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, xarray as xr, joblib, rioxarray, geopandas as gpd
from rasterio.features import geometry_mask
import config as C
M=joblib.load("outputs/model.joblib"); model,le,feat,med=M["model"],M["le"],M["feat"],M["medians"]
st=xr.open_zarr("outputs/feature_stack.zarr")["features"].rio.write_crs(C.UTM)
dist=gpd.read_file("data/boundary/Chitradurga.shp").to_crs(C.UTM)
H,W=st.sizes["y"],st.sizes["x"]; tr=st.rio.transform()
inside=~geometry_mask(dist.geometry,out_shape=(H,W),transform=tr)
cls=np.full((H,W),255,np.uint8); conf=np.zeros((H,W),np.float32); ent=np.zeros((H,W),np.float32)
T=512
for i in range(0,H,T):
    for j in range(0,W,T):
        blk=st.isel(y=slice(i,i+T),x=slice(j,j+T)).transpose("y","x","feature").values
        ok=inside[i:i+T,j:j+T]&np.isfinite(blk).sum(-1).astype(bool)
        if not ok.any(): continue
        Xb=blk[ok]; Xb=np.where(np.isfinite(Xb),Xb,np.array([med[f] for f in feat])[None])
        p=model.predict_proba(Xb)
        sub=cls[i:i+T,j:j+T]; sub[ok]=p.argmax(1)
        conf[i:i+T,j:j+T][ok]=p.max(1); ent[i:i+T,j:j+T][ok]=-(p*np.log(p+1e-9)).sum(1)/np.log(p.shape[1])
def save(a,name,dtype,nodata):
    xr.DataArray(a,dims=("y","x"),coords={"y":st.y,"x":st.x}).rio.write_crs(C.UTM).rio.write_nodata(nodata)\
      .astype(dtype).rio.to_raster(f"outputs/{name}.tif",compress="deflate")
save(cls,"crop_class","uint8",255); save(conf,"confidence","float32",0); save(ent,"entropy","float32",0)
px_ha=C.RES**2/1e4; valid=cls!=255
rows=[dict(cls=c,ha=float((cls==k).sum()*px_ha)) for k,c in enumerate(le.classes_)]
t=pd.DataFrame(rows); t["km2"]=t.ha/100; t["pct_of_valid_classified"]=100*t.ha/t.ha.sum()
t["mean_conf"]=[float(conf[cls==k].mean()) if (cls==k).any() else np.nan for k in range(len(le.classes_))]
t.to_csv("outputs/area_stats.csv",index=False)
tot=inside.sum()*px_ha
pd.Series(dict(district_ha=float(tot),valid_classified_ha=float(valid.sum()*px_ha),
               masked_nodata_ha=float((inside&~valid).sum()*px_ha),low_conf_lt0p5_ha=float(((conf<0.5)&valid).sum()*px_ha))).to_csv("outputs/area_summary.csv")
print(t)
