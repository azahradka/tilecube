import io

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import morecantile
import numpy as np
import xarray as xr
from PIL import Image
from flask import Flask, send_file

import tilecube
from tilecube.ext.flask import tilemap

app = Flask(__name__)
ds = xr.tutorial.open_dataset("air_temperature")
# ds = xr.tutorial.open_dataset("rasm")
print(ds)
x_name = 'lon'
y_name = 'lat'
da_name = 'air'
# ds_name = 'Tair'
# x_name = 'xc'
# y_name = 'yc'
ds[x_name] = ds[x_name] - 180
da = ds.isel(time=0)[da_name]
# da = ds.isel(time=0)['Tair']
cmap = plt.get_cmap('viridis').copy()
cmap.set_under(alpha=0.)
cmap.set_bad(alpha=0.)
bounds = np.linspace(start=float(da.min()), stop=float(da.max()), num=6)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
src_pyproj = None # CRS.from_proj4('+ellps=WGS84 +proj=stere +lat_0=90 +lon_0=252.0 +x_0=0.0 +y_0=0.0 +lat_ts=60 +no_defs')

tilecube = tilecube.from_grid(da[y_name], da[x_name], src_pyproj)


@app.route('/')
def root():
    return "Success"


@app.route('/<int:time>/<int:z>/<int:x>/<int:y>.png')
@tilemap(tilecube, 'bilinear', cmap, norm)
def test(bounds, time):
    (ymin, ymax, xmin, xmax) = bounds
    ds_subset = ds.isel(time=time)[da_name][ymin:ymax, xmin:xmax]
    return ds_subset.data


if __name__ == '__main__':
    app.run(threaded=False, processes=5, debug=True)