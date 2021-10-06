import typing as t
import logging

import numpy as np
import pyproj
from morecantile import Tile
from tqdm import tqdm

from tilecube import distributed
from tilecube.core import TilerFactory, Tiler
from tilecube.storage import TileCubeStorage


def from_grid(y: np.array, x: np.array, proj: pyproj.CRS = None, storage: TileCubeStorage = None):
    tiler_factory = TilerFactory(y, x, proj)
    tc = TileCube(tiler_factory)
    if storage is not None:
        tc.save(storage)
    return tc


def from_storage(storage: TileCubeStorage):
    tiler_factory = storage.read_tiler_factory()
    tc = TileCube(tiler_factory, storage)
    return tc


class TileCube:

    def __init__(self, tiler_factory: TilerFactory, storage: TileCubeStorage = None):
        """ Initialize a `TileCube` from a raw `core.tiler_factory` object

        Most use cases are better served by creating using `tilecube.from_grid`

        tiler_factory:
        storage: `TileCubeStorage` object to use to persist the regridding weights and metadata. Leave None to
                    avoid persistence and operate on-the-fly.

        """
        self.tiler_factory = tiler_factory
        self.storage = storage

    def __repr__(self):
        s = f'TileCube object with input grid dimensions: ' \
            f'y={self.tiler_factory.src_y.shape}, ' \
            f'x={self.tiler_factory.src_x.shape}\n'
        if self.storage is not None:
            s += 'References storage object: \n\t'
            s += str(self.storage).replace('\n', '\n\t')
        return s

    def save(self, storage: TileCubeStorage):
        self.storage = storage
        storage.write_tiler_factory(self.tiler_factory)

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
            if tile_index is True:
                tiler = self.storage.read_tiler(tile)
            # Check the tile index to see if the current tile has been stored and is available. If so, we can reuse it.
            else:
                return None
        else:
            # The tile needs to be generated
            tiler = self.tiler_factory.generate_tiler(tile, method)
        return tiler

    def write_tiler(self, tile: Tile, tiler: t.Union[Tiler, None]):
        log = logging.getLogger(__name__)
        if self.storage is None:
            raise RuntimeError('Cannot write Tiler when there is no storage object associated '
                               'with the TileCube')
        if tiler is None:
            # There was no intersect between the source grid and this tile, so set the tile index to False
            log.debug('Writing "False" index.')
            self.storage.write_index(tile, False)
        else:
            # We were able to calculate a tile generator for this tile, so set the tile index to True and write the
            # tile generator to storage.
            log.debug('Writing "True" index.')
            self.storage.write_index(tile, True)
            log.debug('Writing Tiler.')
            self.storage.write_tiler(tiler)

    @staticmethod
    def _determine_zoom_level_tiles_to_calculate(
            z: int,
            parent_tile_indices: np.ndarray
    ) -> t.Tuple[t.List[Tile], t.List[Tile]]:
        """ Determine a list of the tiles to process for a given zoom level.

        These are tiles which are likely to have a spatial intersection with the source data (ie the tile parent one
        zoom level up had an intersection. If the parent tile index isn't calculated, assume there is an intersection.

        :param z: Zoom level for which to calculate the list of tiles
        :param parent_tile_indices: Array of the parent tile indices, if exists
            These are the index values showing the existance of a tile at zoom level z - 1
        :return: List of the tiles which are needed to be calculated at the given zoom level
        """
        # If we don't have a tile index for one zoom level up, we need to calculate all tiles
        if parent_tile_indices is None:
            tiles_to_process = []
            for y in range(2**z):
                for x in range(2**z):
                    tiles_to_process.append(Tile(x, y, z))
            return tiles_to_process, []
        # If we have the parent tile index, make sure the shape of it is as expected
        if parent_tile_indices.shape[0] != parent_tile_indices.shape[1]:
            raise ValueError('`parent_tile_indices` array should be square')
        if parent_tile_indices.shape[0] != 2**(z - 1):
            raise ValueError(f'The shape of `parent_tile_indices` should be {2**(z - 1)} for zoom level {z}, '
                             f'not {parent_tile_indices.shape[0]}')
        # Each parent tile is split into 4.
        # So tile_indices[0:2, 0:2] are from parent[0, 0], tile_indices[2:4, 0:2] are from parent[1, 0], etc.
        tile_indices = np.empty((parent_tile_indices.shape[0] * 2, parent_tile_indices.shape[1] * 2), np.int8)
        tile_indices[::2, ::2] = parent_tile_indices
        tile_indices[1::2, ::2] = parent_tile_indices
        tile_indices[::2, 1::2] = parent_tile_indices
        tile_indices[1::2, 1::2] = parent_tile_indices
        # We want to calculate tiles if the parent tile exists or is unknown.
        needs_calculation = (tile_indices == 1) | (tile_indices == -1)
        # Create matrix of x and y index positions, and filter them to the ones which need calculation
        x = np.vstack([np.arange(tile_indices.shape[0])] * tile_indices.shape[1])
        y = np.vstack([np.arange(tile_indices.shape[1])] * tile_indices.shape[0]).T
        x_needs_calc = x[needs_calculation]
        y_needs_calc = y[needs_calculation]
        tiles_to_calc = [Tile(x, y, z) for (x, y) in zip(x_needs_calc, y_needs_calc)]
        x_to_skip = x[~needs_calculation]
        y_to_skip = y[~needs_calculation]
        tiles_to_skip = [Tile(x, y, z) for (x, y) in zip(x_to_skip, y_to_skip)]
        return tiles_to_calc, tiles_to_skip

    def generate_zoom_level_tilers(self, z: t.Union[int, t.List[int]], method: str, dask_client=None):
        """ Generate and save `Tiler` objects for all possible tiles at a given zoom level

        Attempt to calculate and save to storage a `Tiler` object for all 2**Z tiles at a given web map zoom level.
        For high zoom levels this might take a very long time and use a large amount of storage, It is recommended to
        start with a low zoom level and gradually increase the zoom level.

        If you have `dask.distributed` installed, you can optionally pass a Dask `Client` object to perform the calculations in
        parallel using all available cores on a local machine or Dask cluster. Operation is not threadsafe.

        Args:
            z:
            method:
            dask_client:

        Returns:

        """
        log = logging.getLogger(__name__)
        if self.storage is None:
            raise RuntimeError('Cannot write Tilers when there is no storage object associated '
                               'with the TileCube')
        z_to_process = z
        if type(z_to_process) == int:
            z_to_process = [z_to_process]
        log.debug(f'Starting calculation of {len(z_to_process)} zoom levels.')
        for z in z_to_process:
            log.info(f'Generating Tilers for zoom level {z}.')
            parent_tile_indices = self.storage.read_zoom_level_indices(z - 1)
            tiles_to_process, tiles_to_skip = self._determine_zoom_level_tiles_to_calculate(z, parent_tile_indices)
            for tile in tiles_to_skip:
                self.write_tiler(tile, None)
            if len(tiles_to_process) == 0:
                log.info(f'No Tilers to generate at zoom level {z}.')
                continue
            if dask_client is None:
                for i, tile in enumerate(tqdm(tiles_to_process)):
                    log.info(f'Generating tile {i+1} of {len(tiles_to_process)} ({tile.x}, {tile.y}, {tile.z}).')
                    tiler = self.tiler_factory.generate_tiler(tile, method)
                    self.write_tiler(tile, tiler)
            else:
                # Test the calculation of the first tile locally, to make debugging of potential issues easier
                log.info('Calculating Tile 0 locally.')
                tile0 = tiles_to_process.pop(0)
                tiler0 = self.tiler_factory.generate_tiler(tile0, method)
                self.write_tiler(tile0, tiler0)
                log.info(f'Submitting {len(tiles_to_process)} tiles to Dask client for processing.')
                tilers = distributed.calculate_tilers_distributed(self.tiler_factory, tiles_to_process, method, dask_client)
                log.info(f'Successfully calculated {len(tilers)} Tilers. Writing results.')
                for tile, tiler in tilers:
                    self.write_tiler(tile, tiler)
                log.info(f'Successfully wrote {len(tilers)} Tilers.')
