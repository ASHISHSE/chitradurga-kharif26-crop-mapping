# Chitradurga Kharif-2026 crop mapping MVP

## STATUS (read first)
| Stage | State |
|---|---|
| A. GT profiling/cleaning (`src/01`) | **Run, results in `outputs/`** |
| B-D. Imagery, models, maps, stats (`src/02-04`) | **Code written, syntax-checked, NOT run.** The authoring sandbox had no route to Sentinel-2 sources, so there are no metrics, rasters or area numbers yet. None are quoted anywhere. |
| E. Streamlit app (`app/app.py`) | Written, not run (needs outputs of 02-04) |

## Run (needs internet; ~ several GB download/compute at 20 m)
```
pip install -r requirements.txt
python src/01_gt_profile_clean.py
python src/02_extract_features.py     # Sentinel-2 L2A, Earth Search STAC, Jun-Nov 2026, 20 m
python src/03_train_validate.py       # RF(balanced) vs XGBoost, 3 km spatial-block GroupKFold, seed 42
python src/04_predict_stats.py        # crop_class.tif, confidence.tif, entropy.tif, area_stats.csv
streamlit run app/app.py
```
Expect to debug 02 first (stackstac/zarr details); consider a Dask cluster or a sub-district test run.
Boundary: supplied `Chitradurga.shp` (QGIS layer "DISTRICT_BDY_coastline"; no version/date in metadata - ask provider).

## Key design choices
* Only classes with >=20 GT points are modelled: **Maize, Ragi, Arecanut, Coconut**. Other 11 labels (<=9 pts each) are not mapped.
* Intercrop labels (67 pts, 14 label variants) are too sparse for a separate analysis; use `crop_type` only as metadata.
