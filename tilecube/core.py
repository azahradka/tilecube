import pyproj
import numpy as np

from generators import PyramidGenerator, TileGenerator
from tilecube.storage import TileCubeStorage


def load(storage: TileCubeStorage, readonly=False):
    pyramid = storage.read_pyramid_generator()
    tc = TileCube(pyramid, storage, readonly)
    return tc


def from_grid(x: np.array, y: np.array, proj: pyproj.CRS = None, storage: TileCubeStorage = None):
    pyramid_generator = PyramidGenerator(x, y, proj)
    tc = TileCube(pyramid_generator, storage)
    if storage is not None:
        storage.write_pyramid_generator(pyramid_generator)
    return tc


class TileCube:

    def __init__(self, pyramid_generator: PyramidGenerator, storage: TileCubeStorage, readonly=False):
        self.pyramid_generator = pyramid_generator
        self.storage = storage
        self.readonly = readonly
        self.tiles = dict()
    """
    
    storage: `PyramidStorage` object to use to persist the regridding weights and metadata. Leave None to
                avoid persistence and operate on-the-fly.

    """

    def get_tile_generator(self, tile, method = None, readonly = False, overwrite = False) -> TileGenerator or None:
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
        readonly = readonly or self.readonly

        if self.storage is not None:
            # Check to see if we already have already used an interpolation method for this zoom level.
            # If so make sure we are being consistent and using the same one
            existing_method = self.storage.read_method(tile)
            if existing_method is not None and existing_method != method:
                raise ValueError(f'Cannot create pyramid tile using interpolation method {method}'
                                 f'when existing tiles at zoom level {tile.z} have been created'
                                 f'using interpolation method {existing_method}.')
            # Check the tile index to see if the parent tile has been stored and did not intersect the source data grid.
            # If so, then we know that the child tile won't either and we can short-circuit.
            if self.storage.read_parent_index(tile) is False:
                return None

        if self.storage is not None and not overwrite:
            # Check the tile index to see if the current tile has been stored and did not intersect the source data grid.
            # If so, then we can short-circuit.
            tile_index = self.storage.read_index(tile)
            if tile_index is False:
                return None
            # Check the tile index to see if the current tile has been stored and is available. If so, we can reuse it.
            elif tile_index is True:
                tg = self.storage.read_tile_generator(tile)
                return tg

        # The tile needs to be generated, so calculate it and store it if possible.
        tg = self.pyramid_generator.calculate_tile_generator(tile, method)
        if self.storage is not None and not readonly:
            # There was no intersect between the source grid and this tile, so set the tile index to False
            if tg is None:
                self.storage.write_index(tile, False)
                return None
            # We were able to calculate a tile generator for this tile, so set the tile index to True and write the
            # tile generator to storage.
            self.storage.write_index(tile, True)
            self.storage.write_tile_generator(tg)
        return tg

    def write_tile_generator(self, tile, method, overwrite=False) -> TileGenerator or None:

        # Return the existing tile if one exists and overwrite is False
        if (not overwrite) and self.storage.check_tile_generator_exists(tile):
            return self.get_tile_generator(tile, method)

        # Get the pyramid
        pyramid_tile = self.get_tile_generator(tile, method)
        if pyramid_tile is None:
            self.storage.write_index(tile, False)
            return None
        self.storage.write_index(tile, True)
        self.storage.write_tile_generator(pyramid_tile)
        return pyramid_tile
