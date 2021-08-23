import io

import numpy as np
import xarray as xr
import morecantile
from flask import Flask, send_file
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

from generators import PyramidGenerator

app = Flask(__name__)
ds = xr.tutorial.open_dataset("air_temperature")
x_name = 'lon'
y_name = 'lat'
ds['lon'] = ds['lon'] - 360
da = ds.isel(time=0)['air']
cmap = plt.get_cmap('viridis').copy()
cmap.set_under(alpha=0.)
cmap.set_bad(alpha=0.)
bounds = np.linspace(start=float(da.min()), stop=float(da.max()), num=6)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

src_pyproj = None # CRS.from_proj4('+ellps=WGS84 +proj=stere +lat_0=90 +lon_0=252.0 +x_0=0.0 +y_0=0.0 +lat_ts=60 +no_defs')
pyramid = PyramidGenerator(da[x_name], da[y_name], src_pyproj)


@app.route('/')
def root():
    return "Success"


@app.route('/<int:time>/<int:z>/<int:x>/<int:y>.png')
def test(time, z, x, y):
    tile = morecantile.Tile(x, y, z)
    if tile.z >=7:
        method = 'nearest_s2d'
    else:
        method = 'bilinear'

    pyramid_tile = pyramid.calculate_tile_generator(tile, method)  # TODO variable methods
    if pyramid_tile is None:
        arr = np.empty((PyramidGenerator.TILESIZE, PyramidGenerator.TILESIZE))
        arr[:] = np.nan
    else:
        ds_subset = ds.isel(time=time)['air'].sel({x_name: slice(pyramid_tile.xmin, pyramid_tile.xmax),
                            y_name: slice(pyramid_tile.ymax, pyramid_tile.ymin)})
        ds_out = pyramid_tile.transform(ds_subset)
        arr = ds_out.values
        # print(arr)
        arr = np.flip(arr, axis=0)
    mask = np.isnan(arr)
    arr = np.ma.array(arr, mask=mask)
    img_arr = cmap(norm(arr))
    im = Image.fromarray(np.uint8(img_arr * 255))
    with io.BytesIO() as buffer:
        im.save(buffer, 'PNG')
        return send_file(io.BytesIO(buffer.getvalue()), mimetype='image/png')


if __name__ == '__main__':
    app.run(threaded=False, processes=5, debug=True)