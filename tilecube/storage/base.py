import abc
import math

import morecantile
import numpy as np

from tilecube.generators import PyramidGenerator, TileGenerator


class TileCubeStorage:

    min_pyramid_version = '0.0.1'

    def __init__(self):
        self.index_lengths = {z: 2**z for z in range(0, 19)}

    @staticmethod
    def _get_parent_tile(tile: morecantile.Tile):
        parent_tile = morecantile.Tile(
            math.floor(tile.x / 2),
            math.floor(tile.y / 2),
            tile.z - 1)
        return parent_tile

    @abc.abstractmethod
    def write_pyramid_generator(self, pyramid_generator: PyramidGenerator):
        pass

    @abc.abstractmethod
    def read_pyramid_generator(self) -> PyramidGenerator:
        pass

    @abc.abstractmethod
    def write_index(self, tile: morecantile.Tile, value: bool):
        pass

    @abc.abstractmethod
    def read_index(self, tile: morecantile.Tile) -> bool or None:
        """ Read the tile index to determine a a PyramidTile exists for the given Tile location.

        Args:
            tile: ZXY Tile location

        Returns: True if the PyramidTile exists, False if it does not.
            None if the index is not initialized.

        """
        pass

    @abc.abstractmethod
    def read_parent_index(self, tile: morecantile.Tile) -> bool or None:
        """ Read the tile index to determine a a PyramidTile exists for the *parent* tile location.

        The parent tile is the tile which encompasses the requested tile location, at zoom level `tile.z - 1`

        Args:
            tile: ZXY Tile location

        Returns: True if the PyramidTile exists, False if it does not.
            None if the index is not initialized.

        """
        pass

    @abc.abstractmethod
    def read_zoom_level_indices(self, z: int) -> np.ndarray or None:
        """ Read the tile indices to determine which PyramidTiles exist for the given zoom level.

        Args:
            z: Zoom level

        Returns: Array of tile indices storing the existence of each PyramidTile in the given zoom level
            Array is Float32 with width and length equal to the number of tiles along x and y axis of the zoom level.
              Values are:
                1. if the PyramidTile exists
                0. if the PyrimidTile does not exist (no intersection with source data)
                -1. if the index for a PyramidTile is not initialized (tile has not been calculated)
            `None` is returned if the index array for the given zoom level does not exist yet.

        """
        pass


    @abc.abstractmethod
    def write_method(self, tile: morecantile.Tile, method):
        pass

    @abc.abstractmethod
    def read_method(self, tile: morecantile.Tile) -> str or None:
        pass

    @abc.abstractmethod
    def write_tile_generator(self, tile_generator: 'TileGenerator'):
        pass

    @abc.abstractmethod
    def read_tile_generator(self, tile: morecantile.Tile) -> 'TileGenerator' or None:
        """ Read pyramid file from disk.

        Args:
            tile: ZXY Tile location

        Returns: The requested `PyramidTile`. `None` if it does not exist.

        """
        pass

    @abc.abstractmethod
    def check_tile_generator_exists(self, tile: morecantile.Tile) -> bool:
        pass
