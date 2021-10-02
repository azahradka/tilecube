from typing import List, Tuple

import numpy as np
import pyproj
from morecantile import Tile

from tilecube.core import TilerFactory, Tiler
from tilecube.storage import TileCubeStorage


class TileCube:

    def __init__(self, tiler_factory: TilerFactory, storage: TileCubeStorage = None):
        self.tiler_factory = tiler_factory
        self.storage = storage
        self.tiles = dict()
    """
    
    storage: `TileCubeStorage` object to use to persist the regridding weights and metadata. Leave None to
                avoid persistence and operate on-the-fly.

    """

    @staticmethod
    def load(storage: TileCubeStorage):
        tiler_factory = storage.read_tiler_factory()
        tc = TileCube(tiler_factory, storage)
        return tc

    @staticmethod
    def from_grid(x: np.array, y: np.array, proj: pyproj.CRS = None, storage: TileCubeStorage = None):
        tiler_factory = TilerFactory(x, y, proj)
        tc = TileCube(tiler_factory, storage)
        if storage is not None:
            storage.write_tiler_factory(tiler_factory)
        return tc

    def get_tiler(self, tile: Tile, method: str = None) -> Tiler or None:
        """ Read or calculate a Tiler.

        * If the tiler exists in storage it is read out and returned.
        * If the tiler does not exist in storage but can be calculated, it is calculated and returned.
        * If the tiler cannot be calculated because there is no overlap between the data and tile,
            then None is returned.

        Args:
            tile:
            method:

        Returns: the requested Tiler. None if there is no overlap between the data and `tile`.

        """
        if self.storage is not None:
            # Check to see if we already have already used an interpolation method for this zoom level.
            # If so make sure we are being consistent and using the same one
            existing_method = self.storage.read_method(tile)
            if existing_method is not None and existing_method != method:
                raise ValueError(f'Cannot create tiler using interpolation method {method}'
                                 f'when existing tiles at zoom level {tile.z} have been created'
                                 f'using interpolation method {existing_method}.')
        else:
            if method is None:
                raise ValueError('Method must be supplied when there is no storage object available to the '
                                 'TilerFactory')

        if self.storage is not None:
            # Check the tile index to see if the parent tile has been stored and did not intersect the source data grid.
            # If so, then we know that the child tile won't either and we can short-circuit.
            if self.storage.read_parent_index(tile) is False:
                return None
            # Check the tile index to see if the current tile has been stored and did not intersect the source data grid.
            # If so, then we can short-circuit.
            tile_index = self.storage.read_index(tile)
            if tile_index is False:
                return None
            # Check the tile index to see if the current tile has been stored and is available. If so, we can reuse it.
            elif tile_index is True:
                tiler = self.storage.read_tiler(tile)
                return tiler

        # The tile needs to be generated
        tiler = self.tiler_factory.generate_tiler(tile, method)

    def write_tiler(self, tile: Tile, tiler: Tiler):
        if self.storage is None:
            raise RuntimeError('Cannot write Tiler when there is no storage object associated '
                               'with the TileCube')
        # There was no intersect between the source grid and this tile, so set the tile index to False
        if tiler is None:
            self.storage.write_index(tile, False)
            return
        # We were able to calculate a tile generator for this tile, so set the tile index to True and write the
        # tile generator to storage.
        self.storage.write_index(tile, True)
        self.storage.write_tiler(tiler)

    @staticmethod
    def _determine_zoom_level_tiles_to_calculate(z: int, parent_tile_indices: np.ndarray) -> List[Tile]:
        """ Determine a list of the tiles to process for a given zoom level.

        These are tiles which are likely to have a spatial intersection with the source data (ie the tile parent one
        zoom level up had an intersection. If the parent tile index isn't calculated, assume there is an intersection.

        :param z: Zoom level for which to calculate the list of tiles
        :param parent_tile_indices: Array of the parent tile indices, if exists
        :return: List of the tiles which are needed to be calculated at the given zoom level
        """
        # If we don't have a tile index for one zoom level up, we need to calculate all tiles
        if parent_tile_indices is None:
            tiles_to_process = []
            for parent_y in range(2**z):
                for parent_x in range(2**z):
                    tiles_to_process.append(Tile(parent_x, parent_y, z))
            return tiles_to_process
        # If we have the parent tile index, make sure the shape of it is as expected
        if parent_tile_indices.shape[0] != parent_tile_indices.shape[1]:
            raise ValueError('`parent_tile_indices` array should be square')
        if parent_tile_indices.shape[0] * 2 != 2**z:
            raise ValueError(f'The shape of `parent_tile_indices` should be {2**z} for zoom level {z}, '
                             f'not {parent_tile_indices.shape[0]}')
        # We want to calculate tiles if the parent tile exists or is unknown
        tile_indices = np.empty((parent_tile_indices.shape[0] * 2, parent_tile_indices.shape[1] * 2), np.int8)
        tile_indices[::2, ::2] = parent_tile_indices
        tile_indices[1::2, ::2] = parent_tile_indices
        tile_indices[::2, 1::2] = parent_tile_indices
        tile_indices[1::2, 1::2] = parent_tile_indices
        needs_calculation = (tile_indices == 1) | (tile_indices == -1)
        x = np.vstack([np.arange(tile_indices.shape[0])] * tile_indices.shape[1])
        y = np.vstack([np.arange(tile_indices.shape[1])] * tile_indices.shape[0]).T
        x_needs_calc = x[needs_calculation]
        y_needs_calc = y[needs_calculation]
        tiles = [Tile(x, y, z) for (x, y) in zip(x_needs_calc, y_needs_calc)]
        return tiles

    def generate_zoom_level_tilers(self, z: int, method: str, dask_client=None):
        if self.storage is None:
            raise RuntimeError('Cannot write Tiler when there is no storage object associated '
                               'with the TileCube')
        parent_tile_indices = self.storage.read_zoom_level_indices(z)
        tiles_to_process = self._determine_zoom_level_tiles_to_calculate(z, parent_tile_indices)
        if len(tiles_to_process) == 0:
            return
        if dask_client is None:
            for tile in tiles_to_process:
                tiler = self.tiler_factory.generate_tiler(tile, method)
                self.write_tiler(tile, tiler)
        else:
            # Test the calculation of the first tile locally, to make debugging of potential issues easier
            tile0 = tiles_to_process.pop(0)
            tiler0 = self.tiler_factory.generate_tiler(tile0, method)
            self.write_tiler(tile0, tiler0)
            tilers = self._calculate_tilers_distributed(tiles_to_process, method, dask_client)
            for tile, tiler in tilers:
                self.write_tiler(tile, tiler)

    def _calculate_tilers_distributed(
            self,
            tiles: List[Tile],
            method,
            dask_client
    ) -> List[Tuple[Tile, Tiler or None]]:
        tile_tuples = [(t.x, t.y, t.z) for t in tiles]
        tiler_factory_dict = self.tiler_factory.to_dict()
        future = dask_client.submit(
            TileCube._dask_worker_calculate_tilers,
            tiles=tile_tuples,
            tiler_factory_dict=tiler_factory_dict,
            method=method)
        results = future.result()
        tilers = []
        for tile_index, tg_dict in results:
            tile = Tile(*tile_index)
            if tg_dict is not None:
                tg = Tiler.from_dict(tg_dict)
            else:
                tg = None
            tilers.append((tile, tg))
        return tilers

    @staticmethod
    def _dask_worker_calculate_tilers(
            tiles: List[Tuple[int, int, int]],
            tiler_factory_dict: dict,
            method: str
    ) -> List[Tuple[Tuple[int, int, int], dict or None]]:
        tiler_factory = TilerFactory.from_dict(tiler_factory_dict)
        tilers = []
        for (x, y, z) in tiles:
            tile = Tile(x, y, z)
            tiler = tiler_factory.generate_tiler(tile, method)
            if tiler is not None:
                tiler_dict = tiler.to_dict()
            else:
                tiler_dict = None
            tilers.append(((x, y, z), tiler_dict))
        return tilers
