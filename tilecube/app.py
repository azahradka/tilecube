import io
import os
import tempfile

import pyproj
from pyproj import CRS
import mercantile
import numpy as np
import xarray as xr
from flask import Flask, send_file, abort
from PIL import Image
import matplotlib as mpl
import matplotlib.colors

from app.core import Pyramid
from app.storage import HDF5PyramidStorage

merc_pyproj = CRS.from_epsg(3857)
src_pyproj = CRS.from_proj4('+ellps=WGS84 +proj=stere +lat_0=90 +lon_0=252.0 +x_0=0.0 +y_0=0.0 +lat_ts=60 +no_defs')
wgs84 = CRS.from_epsg(4326)

merc_lonlat_transformer = pyproj.Transformer.from_crs(merc_pyproj, wgs84, always_xy=True)
src_lonlat_tranformer = pyproj.Transformer.from_crs(src_pyproj, wgs84, always_xy=True)
merc_src_transformer = pyproj.Transformer.from_crs(merc_pyproj, src_pyproj, always_xy=True)

ds = xr.open_dataset('hrdps.nc')
# Create lat-long coordinate grid for source data to be used in xesmf reprojection

pyramid_storage = HDF5PyramidStorage('weights.hdf5')
pyramid = Pyramid(ds['x'], ds['y'], src_pyproj, pyramid_storage)  # TODO: read data from storage

app = Flask(__name__)

cmap = mpl.colors.ListedColormap([
    [242 / 255, 246 / 255, 255 / 255, 1],
    [217 / 255, 228 / 255, 255 / 255, 1],
    [179 / 255, 200 / 255, 255 / 255, 1],
    [128 / 255, 170 / 255, 255 / 255, 1],
    [128 / 255, 191 / 255, 255 / 255, 1],
    [128 / 255, 223 / 255, 255 / 255, 1],
    [128 / 255, 255 / 255, 255 / 255, 1],
    [102 / 255, 255 / 255, 179 / 255, 1],
    [191 / 255, 255 / 255, 102 / 255, 1],
    [255 / 255, 255 / 255, 102 / 255, 1],
    [255 / 255, 204 / 255, 102 / 255, 1],
    [255 / 255, 136 / 255, 77 / 255, 1],
    [255 / 255, 64 / 255, 25 / 255, 1],
    [204 / 255, 0 / 255, 0 / 255, 1],
    [128 / 255, 0 / 255, 0 / 255, 1],
], 'ECCCPrecipitation')
cmap.set_under(alpha=0.)
cmap.set_bad('grey')
bounds = [0.0001, 0.1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

td = tempfile.TemporaryDirectory()
os.chdir(td.name)


@app.route('/<int:z>/<int:x>/<int:y>.png')
def test(z, x, y):
    tile = mercantile.Tile(x, y, z)
    if tile.z >=7:
        method = 'nearest_s2d'
    else:
        method = 'bilinear'

    pyramid_tile = pyramid.get_pyramid_tile(tile, method)  # TODO variable methods
    if pyramid_tile is None:
        arr = np.empty((Pyramid.TILESIZE, Pyramid.TILESIZE))
        arr[:] = np.nan
    else:
        ds_subset = ds['Band1'].sel(x=slice(pyramid_tile.xmin, pyramid_tile.xmax),
                                    y=slice(pyramid_tile.ymin, pyramid_tile.ymax))
        ds_out = pyramid_tile.transform(ds_subset)
        arr = ds_out.values
    # mask = np.isnan(arr)
    # arr = np.ma.array(arr, mask=mask)
    img_arr = cmap(norm(arr))

    img_arr = np.flip(img_arr, axis=0)
    im = Image.fromarray(np.uint8(img_arr * 255))
    with io.BytesIO() as buffer:
        im.save(buffer, 'PNG')
        return send_file(io.BytesIO(buffer.getvalue()), mimetype='image/png')


if __name__ == '__main__':
    app.run(threaded=False, processes=5, debug=True)