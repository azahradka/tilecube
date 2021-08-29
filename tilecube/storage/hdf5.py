import math

import h5py
import morecantile
import numpy as np
import pyproj
import scipy
from morecantile import Tile

import tilecube
from generators import PyramidGenerator, TileGenerator
from storage.base import TileCubeStorage


class HDF5TileCubeStorage(TileCubeStorage):

    def __init__(self, filename: str, mode: str = 'a'):
        super().__init__()
        self.filename = filename
        self.file: h5py.File = h5py.File(filename, mode)

    def write_pyramid_generator(self, pyramid_generator: PyramidGenerator):
        if 'src_x' in self.file:
            del self.file['src_x']
        if 'src_y' in self.file:
            del self.file['src_y']
        x_ds = self.file.create_dataset('src_x', (len(pyramid_generator.src_x)), pyramid_generator.src_x.dtype)
        y_ds = self.file.create_dataset('src_y', (len(pyramid_generator.src_y)), pyramid_generator.src_y.dtype)
        x_ds[:] = pyramid_generator.src_x.values
        y_ds[:] = pyramid_generator.src_y.values

        self.file.attrs['src_proj'] = pyramid_generator.src_proj.to_json()
        self.file.attrs['tile_proj'] = pyramid_generator.tile_proj.to_json()

        self.file.attrs['pyramid_version'] = tilecube.__version__

    def _verify_version_compatibility(self, version):
        v = [int(v) for v in version.split('.')]
        mv = [int(v) for v in self.min_pyramid_version.split('.')]
        if len(v) != 3:
            raise RuntimeError(f'The file was written with an invalid version string: {version}.')
        if ((v[0] < mv[0])
                or (v[0] < mv[0] and v[1] < mv[1])
                or (v[0] < mv[0] and v[1] < mv[1] and v[2] < v[2])):
            raise RuntimeError(f'The pyramid file was was written with version {version} but the minimum'
                               f'version which can be read is {self.min_pyramid_version}. Re-pyramid the '
                               f'file to proceed.')

    def read_pyramid_generator(self) -> PyramidGenerator:
        # TODO add y_flipped
        if 'src_x' not in self.file or 'src_y' not in self.file or 'pyramid_version' not in self.file.attrs:
            raise RuntimeError('The file does not contain a valid Pyramid')
        self._verify_version_compatibility(self.file.attrs['pyramid_version'])
        src_proj_json = self.file.attrs['src_proj']
        src_proj = pyproj.CRS.from_json(src_proj_json)
        # tile_proj_json = self.file.attrs['tile_proj']
        # tile_proj = pyproj.CRS.from_json(tile_proj_json)
        pyramid = PyramidGenerator(
            self.file['src_x'],
            self.file['src_y'],
            proj=src_proj)
        return pyramid

    def write_index(self, tile: Tile, value: bool):
        grp = self.file.require_group(f'/{tile.z}')
        if 'index' not in grp:
            grp.create_dataset(
                'index',
                shape=(self.index_lengths[tile.z], self.index_lengths[tile.z]),
                dtype=np.int8,
                fillvalue=-1)
        if value:
            grp['index'][tile.x, tile.y] = 1
        else:
            grp['index'][tile.x, tile.y] = 0

    def read_index(self, tile) -> bool or None:
        if f'/{tile.z}/index' not in self.file:
            return None
        value = self.file[str(tile.z)]['index'][tile.x][tile.y]
        if value == -1:
            return None
        elif value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ValueError(f'Expected index value to be -1, 0, or 1, not {value}.')

    def read_parent_index(self, tile) -> bool or None:
        if str(tile.z - 1) in self.file.keys():
            parent_tile = self._get_parent_tile(tile)
            return self.read_index(parent_tile)

    def write_method(self, tile, method):
        grp = self.file.require_group(f'/{tile.z}')
        grp.attrs['method'] = method

    def read_method(self, tile) -> str or None:
        if f'/{tile.z}' not in self.file:
            return None
        attrs = self.file[str(tile.z)].attrs
        if 'method' in attrs:
            return attrs['method']
        else:
            return None

    def write_tile_generator(self, pyramid_tile: TileGenerator):
        grp = self.file.require_group(f'/{pyramid_tile.tile.z}/{pyramid_tile.tile.x}/{pyramid_tile.tile.y}')
        grp.create_dataset('row', (pyramid_tile.weights.nnz,), dtype='int32', data=pyramid_tile.weights.row)
        grp.create_dataset('col', (pyramid_tile.weights.nnz,), dtype='int32', data=pyramid_tile.weights.col)
        grp.create_dataset('S', (pyramid_tile.weights.nnz,), dtype=pyramid_tile.weights.dtype, data=pyramid_tile.weights.data)
        grp.attrs['shape_in'] = pyramid_tile.shape_in
        grp.attrs['shape_out'] = pyramid_tile.shape_out
        grp.attrs['bounds'] = pyramid_tile.bounds

    def read_tile_generator(self, tile: Tile) -> TileGenerator or None:
        # Check pyramid index
        index_val = self.read_index(tile)
        if index_val is None or index_val is False:
            return None
        # Index indicates pyramid tile exists, check that it does
        if self.check_tile_generator_exists(tile) is False:
            return None
        # Read pyramid file from file
        grp = self.file[str(tile.z)][str(tile.x)][str(tile.y)]
        row = grp['row'][:]
        col = grp['col'][:]
        S = grp['S'][:]
        shape_in = tuple(grp.attrs['shape_in'])
        shape_out = tuple(grp.attrs['shape_out'])
        bounds = grp.attrs['in_bounds']
        weights = scipy.sparse.coo_matrix(
            (S, (row, col)),
            shape=[shape_out[0] * shape_out[1], shape_in[0] * shape_in[1]])
        return TileGenerator(tile, weights, bounds, shape_in, shape_out)

    def check_tile_generator_exists(self, tile: Tile) -> bool:
        if f'/{tile.z}/{tile.x}/{tile.y}' in self.file:
            return True
        else:
            return False