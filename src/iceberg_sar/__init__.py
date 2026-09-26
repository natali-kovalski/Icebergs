"""Iceberg detection and classification in Sentinel-1 SAR imagery off Newfoundland."""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _use_bundled_proj_gdal() -> None:
    """Ignore PROJ/GDAL data dirs from other installs (e.g. PostGIS sets PROJ_LIB system-wide).

    The rasterio/pyproj wheels ship matching data; a foreign proj.db breaks every CRS lookup.
    Only this process's environment is changed. Variables pointing inside the active
    environment (e.g. conda) are kept.
    """
    prefix = Path(sys.prefix).resolve()
    for var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        value = os.environ.get(var)
        if value and prefix not in Path(value).resolve().parents:
            del os.environ[var]


_use_bundled_proj_gdal()
