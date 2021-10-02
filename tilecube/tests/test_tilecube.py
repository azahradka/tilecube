import os
import tempfile

import numpy as np
import pytest
from morecantile import Tile

from tilecube.core import TilerFactory
from tilecube.storage.hdf5 import HDF5TileCubeStorage
from tilecube.tilecube import TileCube


@pytest.fixture
def tf():
    lat = np.arange(-60, 60, 0.3)
    lon = np.arange(-120, 120, 0.4)
    tf = TilerFactory(lat, lon)
    return tf


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


def test_generate_zoom_level_requires_storage(tf:TilerFactory):
    tc = TileCube(tf)
    with pytest.raises(RuntimeError):
        tc.generate_zoom_level_tilers(0, 'bilinear')


def test_generate_zoom_level_local(tf:TilerFactory):
    method = 'bilinear'
    z = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = HDF5TileCubeStorage(os.path.join(tmpdir, 'test_generate_zoom_level_local.hdf5'))
        tc = TileCube(tf, storage)
        tc.generate_zoom_level_tilers(z, method)
        assert list(tc.storage.file) == ['0']
        assert list(tc.storage.file['0']) == ['0', 'index']
        assert list(tc.storage.file['0']['0']['0']) == ['S', 'col', 'row']
        tc.generate_zoom_level_tilers(1, method)
        assert list(tc.storage.file) == ['0', '1']
        assert list(tc.storage.file['1']) == ['0', '1', 'index']
        assert list(tc.storage.file['1']['0']) == ['0', '1']
        assert list(tc.storage.file['1']['1']['0']) == ['S', 'col', 'row']


