Chitradurga Kharif 2026 Crop Mapping MVP: Write-up

Scope: model-based estimates for the area covered by the GT points (about 10% of the district). Not district-wide figures, not official statistics. Items marked [FILL] must be copied from outputs/ (05_make_report.py fills them automatically).

1. Pipeline

GT Excel -> cleaning (01) -> Sentinel-2 L2A monthly median composites, Jun-Oct 2026 (Nov not yet available), SCL cloud/shadow mask, [FILL: 20/30] m grid, 1-month gap fill (02) -> 7 bands + NDVI, EVI, NDWI, NDRE, NDMI per month + NDVI max/mean/std/amplitude/peak month (03) -> Random Forest vs XGBoost, 3 km spatial-block CV, seed 42 -> classified map, confidence and entropy rasters, area table (04) -> Streamlit app (map, confidence toggle, area table, click query).

2. Key decisions and trade-offs
Sentinel-2 only: fastest to build; SAR (Sentinel-1) is the first addition for monsoon cloud gaps.
Pixel-based: no field boundaries available; objects would reduce noise.
Monthly composites, coarser pixels: fewer cloud gaps and lower compute, at the cost of phenology detail.
Classical ML: about 280 labelled pixels is too few for deep learning; RF/XGBoost are cheap and interpretable.
Spatial CV: stricter than random CV, avoids inflated accuracy from nearby correlated points.
Streamlit on static GeoTIFFs: quick to deliver; no live inference or versioning.
3. Results

GT: 323 rows, 1 duplicate removed, all 322 points inside the district. Label fixes: coconut/Coconut, Dragon, Horse, Promogrante (assumed Pomegranate). Only 4 classes have >= 20 samples and are mapped: Maize 99, Ragi 74, Arecanut 66, Coconut 46. The other 11 classes (<= 9 points each) are not mapped, and no result is claimed for them. GT gaps: points cover about 831 km² (9.8% of 8,435 km²), in two clusters. The rest of the district has no GT.

Area (valid classified pixels inside the processed GT box):

Class	ha	km²	% of valid classified
Arecanut	51,129	511.3	21.4
Coconut	30,856	308.6	12.9
Maize	130,231	1,302.3	54.5
Ragi	26,722	267.2	11.2
Total	238,938	2,389.4	100

Total valid classified area is about 2,389 km² (about 28% of the district), which equals the processed GT box including its buffer, i.e., larger than the 831 km² GT hull. Mean confidence per class is about 0.6 (confirm exact values in area_stats.csv).

Maize takes 54.5% of mapped area although it is 35% of GT points: a sign that non-crop land is being forced into the crop classes. Masked/no-data area reported separately: [FILL] ha. Low-confidence area (max probability < 0.5): Meaning of area numbers: there is no non-crop class and no cropland mask, so every pixel was forced into one of four crops. The table is "area predicted as crop X", not cropped area. It is also not bias-corrected for classification error.

4. Limitations and risks for another season or district

Crop calendars, cloud cover, and labels change, so retraining with fresh GT is needed. Classes with little GT must not be reported. Results come from one season, one sensor and about 10% spatial coverage. Label noise and GPS error on small fields affect training.

5. With two more weeks, more GT or a production mandate

District-wide stratified GT with non-crop classes; cropland mask; Sentinel-1; field-boundary segmentation and field-level reporting; area-adjusted estimates with confidence intervals; multi-season retraining and monitoring; hosted, versioned app/API.

6. Intercropping

Not attempted. Only 67 intercrop points, with 14 spelling variants of the companion crop, no row-level information and mixed pixels at [FILL: 20/30] m. Needed: field labels with companion-crop fractions, <= 10 m imagery, and a separate validation set.
