import morecantile
import numpy as np
import pyproj
import pytest
import xesmf as xe
import xarray as xr

import tilecube
from tilecube.core.tilecube import TileCube
from tilecube.core.pyramid import Pyramid, PyramidTile


def test_pyramid_init():
    lon = np.arange(-120, 120, 0.4)
    lat = np.arange(-60, 60, 0.3)
    pyramid = Pyramid(lon, lat)
    assert pyramid.src_proj.equals(pyproj.CRS.from_epsg(3857))
    assert pyramid.src_bounding_poly.geom_type == 'Polygon'
    assert pytest.approx(pyramid.src_bounding_poly.bounds[0], -119.8)
    assert pytest.approx(pyramid.src_bounding_poly.bounds[1], -59.85)
    assert pytest.approx(pyramid.src_bounding_poly.bounds[2], 119.8)
    assert pytest.approx(pyramid.src_bounding_poly.bounds[3], 59.85)
    assert len(pyramid.src_bounding_poly.exterior.coords) == (256 * 4)
    assert pytest.approx(pyramid.src_resolution, 0.599)


def test_calculate_pyramid_tile():
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    ds['lon'] = ds['lon'] - 360
    pyramid = Pyramid(ds['lon'], ds['lat'])
    tile = morecantile.Tile(0, 0, 1)
    ptile = pyramid.calculate_pyramid_tile(tile, 'bilinear')
    ds_subset = ds.air.sel(lon=slice(ptile.xmin, ptile.xmax),
                           lat=slice(ptile.ymax, ptile.ymin))
    ds_out = ptile.transform(ds_subset)
    import matplotlib.pyplot as plt
    ds_out.plot()
    plt.show()
    pass
