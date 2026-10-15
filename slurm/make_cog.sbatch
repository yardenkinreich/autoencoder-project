#!/bin/bash
#SBATCH --job-name=make-cog
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# Turn a full-size mosaic into a Cloud-Optimized GeoTIFF for QGIS.
#
# The mosaics this project writes are float32, LZW, and STRIPED with
# blockysize=1 - one 109,164-px-wide row per block. QGIS at low zoom has to
# touch every row to draw anything, so a 19 GB file is effectively unusable
# even after it is copied locally. A COG fixes both halves of that: 512x512
# internal tiles, plus internal overview pyramids so a zoomed-out view reads a
# small decimated level instead of the full raster.
#
# uint8 is the default because this is for LOOKING at the mosaic. It cuts
# 19 GB -> ~2 GB and QGIS opens it instantly. Pass DTYPE=float32 to keep the
# real residual values (bigger, slower, but analysable).
#
# Usage:
#   sbatch slurm/make_cog.sbatch <input.tif> <output.tif> [DTYPE] [LO_PCT] [HI_PCT]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_JOB_ID

SRC="${1:?usage: sbatch slurm/make_cog.sbatch <in.tif> <out.tif> [dtype] [lo_pct] [hi_pct]}"
DST="${2:?missing output path}"
DTYPE="${3:-uint8}"
LO_PCT="${4:-0.1}"
HI_PCT="${5:-99.9}"

echo "src   : $SRC"
echo "dst   : $DST"
echo "dtype : $DTYPE  (stretch $LO_PCT - $HI_PCT pct)"
mkdir -p "$(dirname "$DST")"

export GDAL_CACHEMAX=4096
export GDAL_NUM_THREADS=ALL_CPUS

if [ "$DTYPE" = "float32" ]; then
  # Values preserved exactly. PREDICTOR=3 is the floating-point predictor.
  pixi run -e default gdal_translate \
    -of COG \
    -co COMPRESS=DEFLATE -co PREDICTOR=3 -co LEVEL=6 \
    -co BLOCKSIZE=512 -co OVERVIEWS=AUTO -co NUM_THREADS=ALL_CPUS \
    -co BIGTIFF=YES \
    "$SRC" "$DST"
else
  # Percentiles come from a decimated read, not the full raster: 4e9 pixels is
  # far more than a percentile needs, and the sentinel/nodata magnitude check
  # (|v| >= 1e30) is the same one build_catalog and norm_stats use - plain
  # isfinite misses EMP's -3.4e38 fill, which would drag the stretch to junk.
  read -r LO HI < <(pixi run -e default python - "$SRC" "$LO_PCT" "$HI_PCT" <<'PY'
import sys
import numpy as np
import rasterio

path, lo_pct, hi_pct = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with rasterio.open(path) as s:
    # ~10M sample pixels regardless of source size
    step = max(1, int(np.sqrt((s.height * s.width) / 10_000_000)))
    a = s.read(1, out_shape=(s.height // step, s.width // step)).astype(np.float32)
    nod = s.nodata

good = np.isfinite(a) & (np.abs(a) < 1e30)
if nod is not None and np.isfinite(nod):
    good &= a != np.float32(nod)
v = a[good]
lo, hi = np.percentile(v, [lo_pct, hi_pct])
if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    lo, hi = float(v.min()), float(v.max())
print(f"{lo:.10g} {hi:.10g}")
print(f"sampled {v.size:,} valid of {a.size:,} (step {step})", file=sys.stderr)
print(f"stretch {lo:.6g} .. {hi:.6g}", file=sys.stderr)
PY
)
  echo "scale: $LO -> 1, $HI -> 255   (0 reserved for nodata)"
  pixi run -e default gdal_translate \
    -of COG -ot Byte \
    -scale "$LO" "$HI" 1 255 \
    -a_nodata 0 \
    -co COMPRESS=DEFLATE -co PREDICTOR=2 -co LEVEL=6 \
    -co BLOCKSIZE=512 -co OVERVIEWS=AUTO -co NUM_THREADS=ALL_CPUS \
    -co BIGTIFF=YES \
    "$SRC" "$DST"
fi

echo "=== result ==="
ls -lh "$DST"
pixi run -e default gdalinfo "$DST" | grep -iE "^Size|Block|Overviews|COMPRESSION|LAYOUT|Type=" | head
echo "EXIT=0"
