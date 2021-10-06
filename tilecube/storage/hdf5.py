import h5py
import numpy as np
import pyproj
import scipy
from morecantile import Tile

import tilecube
from tilecube.core import TilerFactory, Tiler
from tilecube.storage.base import TileCubeStorage


class HDF5TileCubeStorage(TileCubeStorage):

    def __init__(self, filename: str, mode: str = 'a'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.file: h5py.File = None

    def __repr__(self):
        if self.file is None:
            return f'Closed HDF5TileCubeStorage referencing: {self.filename}\n'
        self._verify_file_open()
        super_str = super().__repr__()
        s = f'HDF5TileCubeStorage referencing: {self.filename}\n'
        s += super_str
        return s

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        self.file: h5py.File = h5py.File(self.filename, self.mode)

    def close(self):
        self.file.close()
        self.file = None

    def _verify_file_open(self):
        if self.file is None:
            raise RuntimeError(f'HDF5 File {self.filename} needs to be open to read/write.')
        try:
            _ = self.file.mode
        except ValueError as e:
            raise RuntimeError(f'HDF5 File {self.filename} needs to be open to read/write.') from e

    def write_tiler_factory(self, tiler_factory: TilerFactory):
        self._verify_file_open()
        if 'src_x' in self.file:
            del self.file['src_x']
        if 'src_y' in self.file:
            del self.file['src_y']
        x_ds = self.file.create_dataset('src_x', tiler_factory.src_x.shape, tiler_factory.src_x.dtype)
        y_ds = self.file.create_dataset('src_y', tiler_factory.src_y.shape, tiler_factory.src_y.dtype)
        x_ds[:] = tiler_factory.src_x.values
        y_ds[:] = tiler_factory.src_y.values

        self.file.attrs['src_proj'] = tiler_factory.src_proj.to_json()
        self.file.attrs['tile_proj'] = tiler_factory.tile_proj.to_json()

        self.file.attrs['tilecube_version'] = tilecube.__version__

    def _verify_version_compatibility(self, version):
        v = [int(v) for v in version.split('.')]
        mv = [int(v) for v in self.min_tilecube_version.split('.')]
        if len(v) != 3:
            raise RuntimeError(f'The file was written with an invalid version string: {version}.')
        if ((v[0] < mv[0])
                or (v[0] < mv[0] and v[1] < mv[1])
                or (v[0] < mv[0] and v[1] < mv[1] and v[2] < v[2])):
            raise RuntimeError(f'The tile_factory file was was written with version {version} but the minimum'
                               f'version which can be read is {self.min_tilecube_version}. Re-tile_factory the '
                               f'file to proceed.')

    def read_tiler_factory(self) -> TilerFactory:
        self._verify_file_open()
        # TODO add y_flipped
        if 'src_x' not in self.file or 'src_y' not in self.file or 'tilecube_version' not in self.file.attrs:
            raise RuntimeError('The file does not contain a valid TileCube')
        self._verify_version_compatibility(self.file.attrs['tilecube_version'])
        src_proj_json = self.file.attrs['src_proj']
        src_proj = pyproj.CRS.from_json(src_proj_json)
        # tile_proj_json = self.file.attrs['tile_proj']
        # tile_proj = pyproj.CRS.from_json(tile_proj_json)
        tiler_factory = TilerFactory(
            self.file['src_x'],
            self.file['src_y'],
            proj=src_proj)
        return tiler_factory

    def write_index(self, tile: Tile, value: bool):
        self._verify_file_open()
        grp = self.file.require_group(f'/{tile.z}')
        if 'index' not in grp:
            grp.create_dataset(
                'index',
                shape=(self._index_length(tile.z), self._index_length(tile.z)),
                dtype=np.int8,
                fillvalue=-1)
        if value:
            grp['index'][tile.x, tile.y] = 1
        else:
            grp['index'][tile.x, tile.y] = 0

    def read_index(self, tile) -> bool or None:
        self._verify_file_open()
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
        self._verify_file_open()
        if str(tile.z - 1) in self.file.keys():
            parent_tile = self._get_parent_tile(tile)
            return self.read_index(parent_tile)

    def read_zoom_level_indices(self, z: int) -> np.ndarray or None:
        self._verify_file_open()
        if f'/{z}/index' not in self.file:
            return None
        indices = self.file[str(z)]['index']
        return indices[()]

    def write_method(self, tile, method):
        self._verify_file_open()
        grp = self.file.require_group(f'/{tile.z}')
        grp.attrs['method'] = method

    def read_method(self, tile) -> str or None:
        self._verify_file_open()
        if f'/{tile.z}' not in self.file:
            return None
        attrs = self.file[str(tile.z)].attrs
        if 'method' in attrs:
            return attrs['method']
        else:
            return None

    def write_tiler(self, tiler: Tiler):
        self._verify_file_open()
        grp_name = f'/{tiler.tile.z}/{tiler.tile.x}/{tiler.tile.y}'
        if grp_name in self.file:
            del self.file[grp_name]
        grp = self.file.create_group(grp_name)
        grp.create_dataset('row', (tiler.weights.nnz,), dtype='int32', data=tiler.weights.row)
        grp.create_dataset('col', (tiler.weights.nnz,), dtype='int32', data=tiler.weights.col)
        grp.create_dataset('S', (tiler.weights.nnz,), dtype=tiler.weights.dtype, data=tiler.weights.data)
        grp.attrs['shape_in'] = tiler.shape_in
        grp.attrs['shape_out'] = tiler.shape_out
        grp.attrs['bounds'] = tiler.bounds

    def read_tiler(self, tile: Tile) -> Tiler or None:
        self._verify_file_open()
        # Check tile_factory index
        index_val = self.read_index(tile)
        if index_val is None or index_val is False:
            return None
        # Index indicates tile_factory tile exists, check that it does
        if self.check_tiler_exists(tile) is False:
            return None
        # Read tile_factory file from file
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
        return Tiler(tile, weights, bounds, shape_in, shape_out)

    def check_tiler_exists(self, tile: Tile) -> bool:
        self._verify_file_open()
        if f'/{tile.z}/{tile.x}/{tile.y}' in self.file:
            return True
        else:
            return False