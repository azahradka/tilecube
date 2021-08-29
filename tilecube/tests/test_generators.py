import morecantile
import numpy as np
import pyproj
import pytest
import scipy.sparse

from generators import PyramidGenerator, TileGenerator


@pytest.fixture
def pg():
    lon = np.arange(-120, 120, 0.4)
    lat = np.arange(-60, 60, 0.3)
    pg = PyramidGenerator(lon, lat)
    return pg


def test_pyramid_init(pg):
    assert pg.src_proj.equals(pyproj.CRS.from_epsg(4326))
    assert pg.src_bounding_poly.geom_type == 'Polygon'
    assert pytest.approx(pg.src_bounding_poly.bounds[0], -119.8)
    assert pytest.approx(pg.src_bounding_poly.bounds[1], -59.85)
    assert pytest.approx(pg.src_bounding_poly.bounds[2], 119.8)
    assert pytest.approx(pg.src_bounding_poly.bounds[3], 59.85)
    assert len(pg.src_bounding_poly.exterior.coords) == (256 * 4)
    assert pytest.approx(pg.src_resolution, 0.599)

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
def test_calculate_pyramid_tile(method, z, x, y):
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
        "bilinear_3_1_1": (5, 19),
        "nearest_s2d_3_1_1": (5, 19),
        "nearest_d2s_3_1_1": (5, 19),
        "patch_3_1_1": (5, 19)}
    tile = morecantile.Tile(x, y, z)
    pyramid = PyramidGenerator(x_grid, y_grid)
    ptile = pyramid.calculate_tile_generator(tile, method)
    assert ptile is not None
    data_subset = data[np.ix_(
        ((y_grid >= ptile.ymin) & (y_grid <= ptile.ymax)),
         ((x_grid >= ptile.xmin) & (x_grid <= ptile.xmax)).reshape(-1)
    )]
    assert data_subset.shape == expected_data_subset_dims[f'{method}_{z}_{x}_{y}']
    tile = ptile.transform(data_subset)
    np.testing.assert_array_equal(tile, expected_results)
    # If things change, you may want to visually re-inspect the results and freeze the expected outputs using:
    #     import matplotlib.pyplot as plt
    #     plt.imshow(tile)
    #     plt.show()
    #     np.save(f'tests/test_data/expected_{method}_{z}_{x}_{y}', tile)
    #     plt.close()
    #     # Use this to write a temp file to generate the dict of expected dimensions, then copy to code
    #     with open('tests/test_data/expected_data_subset_dims.txt', 'a') as f:
    #         f.writelines(f'"{method}_{z}_{x}_{y}": {data_subset.shape},\n')


def test_pyramid_generator_to_from_dict(pg):
    d = pg.to_dict()
    assert len(d) == 3
    assert d['proj'] == '+proj=longlat +datum=WGS84 +no_defs +type=crs'
    assert isinstance(d['x'], np.ndarray)
    pg2 = PyramidGenerator.from_dict(d)
    assert isinstance(pg2, PyramidGenerator)
    assert isinstance(pg2.src_x, np.ndarray)
    assert pg2.src_proj.to_proj4() == pg.src_proj.to_proj4()


def test_tile_generator_to_from_dict(pg: PyramidGenerator):
    tg = pg.calculate_tile_generator(morecantile.Tile(0, 0, 1), 'bilinear')
    d = tg.to_dict()
    assert isinstance(d, dict)
    assert len(d) == 6
    assert 'weights' in d
    tg2 = TileGenerator.from_dict(d)
    assert isinstance(tg2, TileGenerator)
    assert isinstance(tg2.weights, scipy.sparse.coo_matrix)
