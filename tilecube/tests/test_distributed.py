import os
import tempfile

import pytest
from dask import distributed as dd
from morecantile import Tile
import numpy as np

from tilecube import TileCube
from tilecube import distributed
from tilecube.core import TilerFactory, Tiler
from tilecube.storage import HDF5TileCubeStorage


@pytest.fixture(scope='module')
def dask_client():
    client = dd.Client()
    yield client
    client.close()


@pytest.fixture
def tf():
    lat = np.arange(-60, 60, 0.3)
    lon = np.arange(-120, 120, 0.4)
    tf = TilerFactory(lat, lon)
    return tf


def test_dask_worker_calculate_tile_generators_local(tf: TilerFactory):
    method = 'bilinear'
    tiles = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    tf_dict = tf.to_dict()
    results = distributed.dask_worker_calculate_tilers(tiles, tf_dict, method)
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    assert isinstance(results[0][1], dict)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None
    tiler = Tiler.from_dict(results[0][1])
    assert isinstance(tiler, Tiler)


def test_dask_worker_calculate_tile_generators(tf: TilerFactory, dask_client):
    method = 'bilinear'
    tiles = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    tf_dict = tf.to_dict()
    future = dask_client.submit(
        distributed.dask_worker_calculate_tilers,
        tiles=tiles,
        tiler_factory_dict=tf_dict,
        method=method)
    results = future.result()
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    assert isinstance(results[0][1], dict)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None
    tiler = Tiler.from_dict(results[0][1])
    assert isinstance(tiler, Tiler)


def test_calculate_tilers_distributed(tf: TilerFactory, dask_client):
    method = 'bilinear'
    tiles = [
        Tile(0, 0, 0),
        Tile(0, 1, 0),
        Tile(1, 0, 0),
        Tile(1, 1, 0),
    ]
    tc = TileCube(tf)
    results = distributed.calculate_tilers_distributed(tc.tiler_factory, tiles, method, dask_client)
    assert len(results) == 4
    assert results[0][0] == tiles[0]
    # assert isinstance(results[0][1], Tiler)  TODO: Why is this failing??!
    assert results[0][1].shape_in == (400, 600)
    assert results[1][0] == tiles[1]
    assert results[1][1] is None


def test_generate_zoom_level_dask(tf:TilerFactory, dask_client):
    method = 'bilinear'
    z = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with HDF5TileCubeStorage(os.path.join(tmpdir, 'test_generate_zoom_level_local.hdf5')) as storage:
            tc = TileCube(tf, storage)
            tc.generate_zoom_level_tilers(z, method, dask_client=dask_client)
            assert list(tc.storage.file) == ['0']
            assert list(tc.storage.file['0']) == ['0', 'index']
            assert list(tc.storage.file['0']['0']['0']) == ['S', 'col', 'row']
            tc.generate_zoom_level_tilers(1, method, dask_client=dask_client)
            assert list(tc.storage.file) == ['0', '1']
            assert list(tc.storage.file['1']) == ['0', '1', 'index']
            assert list(tc.storage.file['1']['0']) == ['0', '1']
            assert list(tc.storage.file['1']['1']['0']) == ['S', 'col', 'row']