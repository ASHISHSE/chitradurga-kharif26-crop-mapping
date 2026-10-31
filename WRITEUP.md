# Chitradurga Kharif 2026 crop mapping - interim write-up
**Status: Task A complete. Tasks B-E are implemented as code but not yet executed (no imagery access in the build environment). No accuracy or area figures exist yet; they must come from running the pipeline.**

## 1. Pipeline
GT Excel -> clean (`01`) -> Sentinel-2 L2A monthly median composites Jun-Nov 2026, SCL cloud/shadow mask, 20 m, 10 bands + NDVI/EVI/NDWI/NDRE/NDMI + NDVI max/mean/std/amplitude/peak month (`02`) -> RF vs XGBoost with spatial-block CV (`03`) -> tiled prediction, probability/entropy rasters, area table (`04`) -> Streamlit app.

## 2. GT findings (actual)
* 323 rows, 5 columns (Latitude, Longitude, Crop, Crop type, Inter crop 1); no missing coordinates; 1 exact duplicate removed -> 322.
* All 322 points lie inside the district; nearest is 11.3 km from the boundary, so edge mixed pixels are not a concern.
* 12 points have a neighbour <20 m (same or adjacent pixel): kept but block-grouped in CV.
* Label inconsistencies: `coconut`/`Coconut` (18 fixed), `Dragon`/`Dragonfruit`, `Horse`/`Horse gram`, `Promogrante` (assumed Pomegranate; need field-team confirmation). Intercrop column has 14 spellings.
* Class counts: Maize 99, Ragi 74, Arecanut 66, Coconut 46, Banana 9, Cotton 7, Onion 5, Papaya 4, Dragonfruit 3, Horse gram 2, Chilli 2, Soybean 2, Basil 1, Pomegranate 1, Marigold 1.
* **Spatial coverage: the GT convex hull is ~831 km2, 9.8% of the 8,435 km2 district.** Points form two clusters (within ~5 km linkage). Predictions elsewhere are extrapolation; the combined map should carry a GT-support/out-of-domain warning, and statistics must not be presented as district-wide estimates.
* **Class scope:** only Maize, Ragi, Arecanut, Coconut (>=20 pts) are modelled. Arecanut and coconut are perennial plantations, so a seasonal time series separates them from Maize/Ragi easily but from each other poorly; expect that confusion.
* **Missing negative class:** GT has no non-crop/fallow/forest/built-up points, so the classifier would label every pixel as one of the four crops. A cropland mask (e.g. ESA WorldCover, or an NDVI-based rule) is needed before area statistics mean anything. Not yet added.

## 3. Decisions (planned)
Sentinel-2 only (SAR optional, deferred); pixel-based (no field boundaries available); monthly composites with at most 1-month linear gap-filling (monsoon cloud); classical ML (322 points, 4 usable classes); 3 km block GroupKFold; class-weighted RF/XGB; Streamlit for speed.

## 4. Not yet done
Run B-E; confirm labels; add cropland mask and out-of-domain flag; per-class metrics, confusion matrix, importance; uncertainty maps; app test; intercrop feasibility note (current evidence: 67 intercrop points, mostly arecanut/coconut with coconut or maize as companion - too sparse and 30 m-class mixing at 20 m resolution for a defensible map).

## 5. With more time/GT
Stratified GT across the whole district (esp. Hagari/Tungabhadra belt and north), non-crop classes, field-boundary segmentation, Sentinel-1 for monsoon cloud, retraining and monitoring per season.
