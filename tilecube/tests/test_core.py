import morecantile
import numpy as np
import pyproj
import pytest
import scipy.sparse

from tilecube.core import TilerFactory, Tiler


def test_tiler_factory_init():
    lat = np.arange(-60, 60, 0.3)
    lon = np.arange(-120, 120, 0.4)
    tf = TilerFactory(lat, lon)
    assert tf.src_proj.equals(pyproj.CRS.from_epsg(4326))
    # TODO: expand


def test_tiler_factory_init_2d_xy():
    import xarray as xr
    ds = xr.tutorial.open_dataset("rasm")
    y = ds['yc']
    x = ds['xc']
    x = x - 180
    tf = TilerFactory(y, x)


@pytest.mark.parametrize('method', [
    "bilinear",
    # Need to figure out handling of lat_b and lon_b cell boundarys for xesmf
    pytest.param("conservative", marks=pytest.mark.xfail),
    "nearest_s2d",
    "nearest_d2s",
    "patch",
])
@pytest.mark.parametrize(('z', 'x', 'y'), [
    (0, 0, 0),
    (1, 0, 0),
    (2, 1, 1),
    (3, 1, 1)])
def test_calculate_tiler(method, z, x, y):
    # This test data originally came from the xarray tutorial dataset "air_temperature"
    # Regenerate using the below:
    #     ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    #     ds['lon'] = ds['lon'] - 360  # lon needs to be centered on zero degrees
    #     np.save('tests/test_data/x', ds['lon'].data)
    #     np.save('tests/test_data/y', ds['lat'].data)
    #     np.save('tests/test_data/data', ds.air.data)
    x_grid = np.load('tests/test_data/x.npy')
    y_grid = np.load('tests/test_data/y.npy')
    data = np.load('tests/test_data/data.npy')
    expected_results = np.load(f'tests/test_data/expected_{method}_{z}_{x}_{y}.npy')
    expected_data_subset_dims = {
        "bilinear_0_0_0": (25, 53),
        "nearest_s2d_0_0_0": (25, 53),
        "nearest_d2s_0_0_0": (25, 53),
        "patch_0_0_0": (25, 53),
        "bilinear_1_0_0": (25, 53),
        "nearest_s2d_1_0_0": (25, 53),
        "nearest_d2s_1_0_0": (25, 53),
        "patch_1_0_0": (25, 53),
        "bilinear_2_1_1": (22, 25),
        "nearest_s2d_2_1_1": (22, 25),
        "nearest_d2s_2_1_1": (22, 25),
        "patch_2_1_1": (22, 25),
        "bilinear_3_1_1": (5, 20),
        "nearest_s2d_3_1_1": (5, 20),
        "nearest_d2s_3_1_1": (5, 20),
        "patch_3_1_1": (5, 20)}
    tile = morecantile.Tile(x, y, z)
    tiler_factory = TilerFactory(y_grid, x_grid)
    tiler = tiler_factory.generate_tiler(tile, method)
    assert tiler is not None
    data_subset = data[tiler.ymin:tiler.ymax, tiler.xmin:tiler.xmax]
    assert data_subset.shape == expected_data_subset_dims[f'{method}_{z}_{x}_{y}']
    tile = tiler.transform(data_subset)
    np.testing.assert_array_equal(tile, expected_results)
    # # If things change, you may want to visually re-inspect the results and freeze the expected outputs using:
    # import matplotlib.pyplot as plt
    # plt.imshow(tile)
    # plt.show()
    # np.save(f'tests/test_data/expected_{method}_{z}_{x}_{y}', tile)
    # plt.close()
    # # Use this to write a temp file to generate the dict of expected dimensions, then copy to code
    # with open('tests/test_data/expected_data_subset_dims.txt', 'a') as f:
    #     f.writelines(f'"{method}_{z}_{x}_{y}": {data_subset.shape},\n')


def test_tiler_factory_to_from_dict():
    lon = np.arange(-120, 120, 0.4)
    lat = np.arange(-60, 60, 0.3)
    tf = TilerFactory(lat, lon)
    d = tf.to_dict()
    assert len(d) == 3
    assert d['proj'] == '+proj=longlat +datum=WGS84 +no_defs +type=crs'
    assert isinstance(d['x'], np.ndarray)
    tf2 = TilerFactory.from_dict(d)
    assert isinstance(tf2, TilerFactory)
    assert isinstance(tf2.src_x, np.ndarray)
    assert tf2.src_proj.to_proj4() == tf.src_proj.to_proj4()


def test_tiler_to_from_dict():
    lon = np.arange(-120, 120, 0.4)
    lat = np.arange(-60, 60, 0.3)
    tf = TilerFactory(lat, lon)
    tiler = tf.generate_tiler(morecantile.Tile(0, 0, 1), 'bilinear')
    d = tiler.to_dict()
    assert isinstance(d, dict)
    assert len(d) == 6
    assert 'weights' in d
    tile2 = Tiler.from_dict(d)
    assert isinstance(tile2, Tiler)
    assert isinstance(tile2.weights, scipy.sparse.coo_matrix)


def test_yxgrid_to_table():
    lat = [20, 40, 60]
    lon = [-100, -50, 0, 50, 100]
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    iy, ix, yv, xv = TilerFactory._yxgrid_to_table(lat_grid, lon_grid)
    iy_expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    ix_expected = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    yv_expected = np.array([20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60])
    xv_expected = np.array([-100, -50, 0, 50, 100, -100, -50, 0, 50, 100, -100, -50, 0, 50, 100])
    np.testing.assert_array_equal(iy, iy_expected)
    np.testing.assert_array_equal(ix, ix_expected)
    np.testing.assert_array_equal(yv, yv_expected)
    np.testing.assert_array_equal(xv, xv_expected)


def test_input_grid_to_2d_with_input_1d_grid():
    lat = np.array([20, 40, 60])
    lon = np.array([-100, -50, 0, 50, 100])
    ygrid, xgrid = TilerFactory._input_grid_to_2d(lat, lon)
    ygrid_expected = np.array([
        [20, 20, 20, 20, 20],
        [40, 40, 40, 40, 40],
        [60, 60, 60, 60, 60],
    ])
    xgrid_expected = np.array([
        [-100, -50, 0, 50, 100],
        [-100, -50, 0, 50, 100],
        [-100, -50, 0, 50, 100],
    ])
    np.testing.assert_array_equal(ygrid, ygrid_expected)
    np.testing.assert_array_equal(xgrid, xgrid_expected)


def test_input_grid_to_2d_with_input_2d_grid():
    ygrid_expected = np.array([
        [20, 20, 20, 20, 20],
        [40, 40, 40, 40, 40],
        [60, 60, 60, 60, 60],
    ])
    xgrid_expected = np.array([
        [-100, -50, 0, 50, 100],
        [-100, -50, 0, 50, 100],
        [-100, -50, 0, 50, 100],
    ])
    ygrid, xgrid = TilerFactory._input_grid_to_2d(ygrid_expected, xgrid_expected)
    np.testing.assert_array_equal(ygrid, ygrid_expected)
    np.testing.assert_array_equal(xgrid, xgrid_expected)


def test_input_grid_to_2d_with_input_invalid_grid():
    lat = [20, 40, 60]
    lon = [-100, -50, 0, 50, 100]
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    lon_grid_wrong_dim = lon_grid[:, :-2]
    with pytest.raises(ValueError):
        TilerFactory._input_grid_to_2d(lat_grid, lon_grid_wrong_dim)
    lon_grid_wrong_number_dim = np.stack([lon_grid, lon_grid], axis=2)
    with pytest.raises(ValueError):
        TilerFactory._input_grid_to_2d(lat_grid, lon_grid_wrong_number_dim)
