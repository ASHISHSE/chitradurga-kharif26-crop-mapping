"""Task E: Streamlit MVP. Run: streamlit run app/app.py  (after pipeline has produced outputs/)"""
import numpy as np, pandas as pd, streamlit as st, rasterio, folium, joblib
from streamlit_folium import st_folium
from pyproj import Transformer
import matplotlib.pyplot as plt
st.set_page_config(layout="wide"); st.title("Chitradurga Kharif 2026 crop map (model-based, not official statistics)")
M=joblib.load("outputs/model.joblib"); names=list(M["le"].classes_)
cls=rasterio.open("outputs/crop_class.tif"); conf=rasterio.open("outputs/confidence.tif")
@st.cache_data
def overlay(show_conf):
    from rasterio.warp import transform_bounds
    step=max(1,cls.width//1500)
    a=cls.read(1)[::step,::step]; c=conf.read(1)[::step,::step]
    pal=np.array([[0.2,0.6,0.2],[0.1,0.4,0.8],[0.9,0.8,0.1],[0.8,0.3,0.2]]+[[0.5,0.5,0.5]]*20)
    rgba=np.zeros(a.shape+(4,)); ok=a!=255
    if show_conf: rgba[ok]=np.c_[plt.cm.RdYlGn(c[ok])[:,:3],np.full(ok.sum(),0.8)]
    else: rgba[ok]=np.c_[pal[a[ok]],np.full(ok.sum(),0.8)]
    return rgba,transform_bounds(cls.crs,"EPSG:4326",*cls.bounds)
show_conf=st.sidebar.toggle("Show confidence layer (max class probability)")
rgba,b=overlay(show_conf)
m=folium.Map(location=[(b[1]+b[3])/2,(b[0]+b[2])/2],zoom_start=9)
folium.raster_layers.ImageOverlay(rgba,bounds=[[b[1],b[0]],[b[3],b[2]]],opacity=1).add_to(m)
L,R=st.columns([2,1])
with L: out=st_folium(m,height=600,width=900)
with R:
    st.subheader("Legend"); [st.markdown(f"- {n}") for n in names]
    t=pd.read_csv("outputs/area_stats.csv"); st.subheader("Area by class"); st.dataframe(t.round(2)); st.bar_chart(t.set_index("cls")["km2"])
    st.caption("Percent = share of valid classified pixels (not of whole district).")
    if out and out.get("last_clicked"):
        lat,lon=out["last_clicked"]["lat"],out["last_clicked"]["lng"]
        x,y=Transformer.from_crs(4326,cls.crs,always_xy=True).transform(lon,lat)
        r,c=cls.index(x,y); k=int(cls.read(1,window=((r,r+1),(c,c+1)))[0,0]); p=float(conf.read(1,window=((r,r+1),(c,c+1)))[0,0])
        st.success(f"Predicted: {names[k] if k<len(names) else 'masked'}  |  confidence {p:.2f}")



