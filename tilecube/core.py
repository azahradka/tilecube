from typing import List, Tuple

import numpy as np
import pyproj
from morecantile import Tile

from tilecube.generators import PyramidGenerator, TileGenerator
from tilecube.storage import TileCubeStorage


class TileCube:

    def __init__(self, pyramid_generator: PyramidGenerator, storage: TileCubeStorage = None):
        self.pyramid_generator = pyramid_generator
        self.storage = storage
        self.tiles = dict()
    """
    
    storage: `PyramidStorage` object to use to persist the regridding weights and metadata. Leave None to
                avoid persistence and operate on-the-fly.

    """

    @staticmethod
    def load(storage: TileCubeStorage):
        pyramid = storage.read_pyramid_generator()
        tc = TileCube(pyramid, storage)
        return tc

    @staticmethod
    def from_grid(x: np.array, y: np.array, proj: pyproj.CRS = None, storage: TileCubeStorage = None):
        pyramid_generator = PyramidGenerator(x, y, proj)
        tc = TileCube(pyramid_generator, storage)
        if storage is not None:
            storage.write_pyramid_generator(pyramid_generator)
        return tc

    def get_tile_generator(self, tile: Tile, method: str = None) -> TileGenerator or None:
        """ Read or calculate a TileGenerator.

        * If the pyramid tile exists in storage it is read out and returned.
        * If the pyramid tile does not exist in storage but can be calculated, it is calculated and returned.
        * If the pyramid tile cannot be calculated because there is no overlap between the data and tile,
            then None is returned.

        Args:
            tile:
            method:

        Returns: the requested PyramidTile. None if there is no overlap between the data and `tile`.

        """
        if self.storage is not None:
            # Check to see if we already have already used an interpolation method for this zoom level.
            # If so make sure we are being consistent and using the same one
            existing_method = self.storage.read_method(tile)
            if existing_method is not None and existing_method != method:
                raise ValueError(f'Cannot create pyramid tile using interpolation method {method}'
                                 f'when existing tiles at zoom level {tile.z} have been created'
                                 f'using interpolation method {existing_method}.')
        else:
            if method is None:
                raise ValueError('Method must be supplied when there is no storage object available to the '
                                 'PyramidGenerator')

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
                tg = self.storage.read_tile_generator(tile)
                return tg

        # The tile needs to be generated
        tg = self.pyramid_generator.calculate_tile_generator(tile, method)

    def write_tile_generator(self, tile: Tile, tile_generator: TileGenerator):
        if self.storage is None:
            raise RuntimeError('Cannot write TileGenerator when there is no storage object associated '
                               'with the TileCube')
        # There was no intersect between the source grid and this tile, so set the tile index to False
        if tile_generator is None:
            self.storage.write_index(tile, False)
            return
        # We were able to calculate a tile generator for this tile, so set the tile index to True and write the
        # tile generator to storage.
        self.storage.write_index(tile, True)
        self.storage.write_tile_generator(tile_generator)

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

    def generate_pyramid_level(self, z: int, method: str, dask_client=None):
        if self.storage is None:
            raise RuntimeError('Cannot write TileGenerator when there is no storage object associated '
                               'with the TileCube')
        parent_tile_indices = self.storage.read_zoom_level_indices(z)
        tiles_to_process = self._determine_zoom_level_tiles_to_calculate(z, parent_tile_indices)
        if len(tiles_to_process) == 0:
            return
        if dask_client is None:
            for tile in tiles_to_process:
                tile_generator = self.pyramid_generator.calculate_tile_generator(tile, method)
                self.write_tile_generator(tile, tile_generator)
        else:
            # Test the calculation of the first tile locally, to make debugging of potential issues easier
            tile0 = tiles_to_process.pop(0)
            tile_generator0 = self.pyramid_generator.calculate_tile_generator(tile0, method)
            self.write_tile_generator(tile0, tile_generator0)
            tile_generators = self._calculate_tile_generators_distributed(tiles_to_process, method, dask_client)
            for tile, tile_generator in tile_generators:
                self.write_tile_generator(tile, tile_generator)

    def _calculate_tile_generators_distributed(
            self,
            tiles: List[Tile],
            method,
            dask_client
    ) -> List[Tuple[Tile, TileGenerator or None]]:
        tile_tuples = [(t.x, t.y, t.z) for t in tiles]
        pg_dict = self.pyramid_generator.to_dict()
        future = dask_client.submit(
            TileCube._dask_worker_calculate_tile_generators,
            tiles=tile_tuples,
            pyramid_generator=pg_dict,
            method=method)
        results = future.result()
        tile_generators = []
        for tile_index, tg_dict in results:
            tile = Tile(*tile_index)
            if tg_dict is not None:
                tg = TileGenerator.from_dict(tg_dict)
            else:
                tg = None
            tile_generators.append((tile, tg))
        return tile_generators

    @staticmethod
    def _dask_worker_calculate_tile_generators(
            tiles: List[Tuple[int, int, int]],
            pyramid_generator: dict,
            method: str
    ) -> List[Tuple[Tuple[int, int, int], dict or None]]:
        pg = PyramidGenerator.from_dict(pyramid_generator)
        tile_generators = []
        for (x, y, z) in tiles:
            tile = Tile(x, y, z)
            tg = pg.calculate_tile_generator(tile, method)
            if tg is not None:
                tg_dict = tg.to_dict()
            else:
                tg_dict = None
            tile_generators.append(((x, y, z), tg_dict))
        return tile_generators
