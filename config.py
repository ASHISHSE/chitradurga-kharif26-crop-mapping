SEED=42
UTM=32643
RES=30                   # metres; 10 m is possible but ~4x compute
DATE_RANGE="2026-06-01/2026-11-30"
STAC_URL="https://earth-search.aws.element84.com/v1"   # Sentinel-2 L2A COGs (open, no key)
S2_BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]
MONTHS=["2026-06","2026-07","2026-08","2026-09","2026-10","2026-11"]
MAX_CLOUD=40
MIN_PER_CLASS=20
CLASSES=["Arecanut","Coconut","Maize","Ragi"]  # classes with n>=MIN_PER_CLASS after cleaning
