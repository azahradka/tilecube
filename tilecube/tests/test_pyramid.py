import morecantile
import numpy as np
import pyproj
import pytest
import xarray as xr

from generators import PyramidGenerator


def test_pyramid_init():
    lon = np.arange(-120, 120, 0.4)
    lat = np.arange(-60, 60, 0.3)
    pyramid = PyramidGenerator(lon, lat)
    assert pyramid.src_proj.equals(pyproj.CRS.from_epsg(4326))
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
    pyramid = PyramidGenerator(ds['lon'].data, ds['lat'].data)
    tile = morecantile.Tile(0, 0, 1)
    ptile = pyramid.calculate_tile_generator(tile, 'bilinear')
    ds_subset = ds.air.sel(lon=slice(ptile.xmin, ptile.xmax),
                           lat=slice(ptile.ymax, ptile.ymin))
    ds_out = ptile.transform(ds_subset.data)
    # TODO finish
    import matplotlib.pyplot as plt
    plt.imshow(ds_out)
    plt.show()
    pass
