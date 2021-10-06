import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import xarray as xr
from flask import Flask

import tilecube
from tilecube.ext.flask import tilemap


app = Flask(__name__)
ds = xr.tutorial.open_dataset("rasm")
ds['xc'] = ds['xc'].where(ds['xc'] < 180, ds['xc'] - 360)
da = ds['Tair']
cmap = plt.get_cmap('RdBu').reversed('BuRd').copy()
norm = mpl.colors.Normalize(da.min(), da.max())

store = tilecube.storage.HDF5TileCubeStorage('rasm_test.hdf5', mode='r')
store.open()
tilecube = tilecube.from_storage(store)


@app.route('/<int:time>/<int:z>/<int:x>/<int:y>.png')
@tilemap(tilecube, 'nearest_s2d', cmap, norm)
def air_temp(bounds, time):
    (ymin, ymax, xmin, xmax) = bounds
    ds_subset = da.isel(time=time)[ymin:ymax, xmin:xmax]
    return ds_subset.data



if __name__ == '__main__':
    app.run(threaded=False, processes=5, debug=True)
