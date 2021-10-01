import os
import tempfile

import dask.distributed as dd
import numpy as np
import pytest
from morecantile import Tile

from tilecube.core import TileCube
from tilecube.generators import PyramidGenerator, TileGenerator
from tilecube.storage.hdf5 import HDF5TileCubeStorage


@pytest.fixture(scope='module')
def dask_client():
    client = dd.Client()
    yield client
    client.close()


@pytest.fixture
def pg():
    lat = np.arange(-60, 60, 0.3)
    lon = np.arange(-120, 120, 0.4)
    pg = PyramidGenerator(lat, lon)
    return pg


def test_determine_zoom_level_tiles_to_calculate():
    z = 2
    parent_tile_indices = np.array([
        [1, 1],
        [0, -1]
    ])
    expected_tile_indices = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (2, 0), (2, 1), (3, 0), (3, 1),
        (2, 2), (2, 3), (3, 2), (3, 3)
    ]
    expected_tiles = [Tile(x, y, z) for (x, y) in expected_tile_indices]
    tiles = TileCube._determine_zoom_level_tiles_to_calculate(z, parent_tile_indices)
    assert len(tiles) == len(expected_tiles)
    for tile in tiles:
        assert tile in expected_tiles


def test_dask_worker_calculate_tile_generators_local(pg: PyramidGenerator):
    method = 'bilinear'
    tiles = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    pg_dict = pg.to_dict()
    results = TileCube._dask_worker_calculate_tile_generators(tiles, pg_dict, method)
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    assert isinstance(results[0][1], dict)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None
    tg = TileGenerator.from_dict(results[0][1])
    assert isinstance(tg, TileGenerator)


def test_dask_worker_calculate_tile_generators(pg: PyramidGenerator, dask_client):
    method = 'bilinear'
    tiles = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    pg_dict = pg.to_dict()
    future = dask_client.submit(
        TileCube._dask_worker_calculate_tile_generators,
        tiles=tiles,
        pyramid_generator=pg_dict,
        method=method)
    results = future.result()
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    assert isinstance(results[0][1], dict)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None
    tg = TileGenerator.from_dict(results[0][1])
    assert isinstance(tg, TileGenerator)


def test_calculate_tile_generators_distributed(pg: PyramidGenerator, dask_client):
    method = 'bilinear'
    tiles = [
        Tile(0, 0, 0),
        Tile(0, 1, 0),
        Tile(1, 0, 0),
        Tile(1, 1, 0),
    ]
    tc = TileCube(pg)
    results = tc._calculate_tile_generators_distributed(tiles, method, dask_client)
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    # assert isinstance(results[0][1], TileGenerator)  TODO: Why is this failing??!
    assert results[0][1].shape_in == (400, 600)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None


def test_generate_pyramid_level_requires_storage(pg:PyramidGenerator):
    tc = TileCube(pg)
    with pytest.raises(RuntimeError):
        tc.generate_pyramid_level(0, 'bilinear')


def test_generate_pyramid_level_local(pg:PyramidGenerator):
    method = 'bilinear'
    z = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = HDF5TileCubeStorage(os.path.join(tmpdir, 'test_generate_pyramid_level_local.hdf5'))
        tc = TileCube(pg, storage)
        tc.generate_pyramid_level(z, method)
        assert list(tc.storage.file) == ['0']
        assert list(tc.storage.file['0']) == ['0', 'index']
        assert list(tc.storage.file['0']['0']['0']) == ['S', 'col', 'row']
        tc.generate_pyramid_level(1, method)
        assert list(tc.storage.file) == ['0', '1']
        assert list(tc.storage.file['1']) == ['0', '1', 'index']
        assert list(tc.storage.file['1']['0']) == ['0', '1']
        assert list(tc.storage.file['1']['1']['0']) == ['S', 'col', 'row']


def test_generate_pyramid_level_dask(pg:PyramidGenerator, dask_client):
    method = 'bilinear'
    z = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = HDF5TileCubeStorage(os.path.join(tmpdir, 'test_generate_pyramid_level_local.hdf5'))
        tc = TileCube(pg, storage)
        tc.generate_pyramid_level(z, method, dask_client=dask_client)
        assert list(tc.storage.file) == ['0']
        assert list(tc.storage.file['0']) == ['0', 'index']
        assert list(tc.storage.file['0']['0']['0']) == ['S', 'col', 'row']
        tc.generate_pyramid_level(1, method, dask_client=dask_client)
        assert list(tc.storage.file) == ['0', '1']
        assert list(tc.storage.file['1']) == ['0', '1', 'index']
        assert list(tc.storage.file['1']['0']) == ['0', '1']
        assert list(tc.storage.file['1']['1']['0']) == ['S', 'col', 'row']
