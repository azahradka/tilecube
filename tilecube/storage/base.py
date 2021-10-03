import abc
import math

import morecantile
import numpy as np

from tilecube.core import TilerFactory, Tiler


class TileCubeStorage:

    min_tilecube_version = '0.0.1'
    MAX_ZOOM_LEVEL = 19
    INDEX_LENGTHS = {z: 2 ** z for z in range(MAX_ZOOM_LEVEL)}

    def __repr__(self):
        s = ''
        for z in range(self.MAX_ZOOM_LEVEL):
            indices = self.read_zoom_level_indices(0)
            if indices is None:
                break
            tiles = np.sum(indices[indices == 1]) + np.sum(indices[indices == -1])
            total = self.INDEX_LENGTHS[z]
            s += f'Z = {z}: {tiles} / {total}\n'
        if s == '':
            s = 'No zoom levels initialized.'
        return s

    @staticmethod
    def _get_parent_tile(tile: morecantile.Tile):
        parent_tile = morecantile.Tile(
            math.floor(tile.x / 2),
            math.floor(tile.y / 2),
            tile.z - 1)
        return parent_tile

    @abc.abstractmethod
    def write_tiler_factory(self, tiler_factory: TilerFactory):
        pass

    @abc.abstractmethod
    def read_tiler_factory(self) -> TilerFactory:
        pass

    @abc.abstractmethod
    def write_index(self, tile: morecantile.Tile, value: bool):
        pass

    @abc.abstractmethod
    def read_index(self, tile: morecantile.Tile) -> bool or None:
        """ Read the tile index to determine a a Tiler exists for the given Tile location.

        Args:
            tile: ZXY Tile location

        Returns: True if the Tiler exists, False if it does not.
            None if the index is not initialized.

        """
        pass

    @abc.abstractmethod
    def read_parent_index(self, tile: morecantile.Tile) -> bool or None:
        """ Read the tile index to determine a Tiler exists for the *parent* tile location.

        The parent tile is the tile which encompasses the requested tile location, at zoom level `tile.z - 1`

        Args:
            tile: ZXY Tile location

        Returns: True if the Tiler exists, False if it does not.
            None if the index is not initialized.

        """
        pass

    @abc.abstractmethod
    def read_zoom_level_indices(self, z: int) -> np.ndarray or None:
        """ Read the tile indices to determine which Tilers exist for the given zoom level.

        Args:
            z: Zoom level

        Returns: Array of tile indices storing the existence of each Tiler in the given zoom level
            Array is Float32 with width and length equal to the number of tiles along x and y axis of the zoom level.
              Values are:
                1. if the Tiler exists
                0. if the PyrimidTile does not exist (no intersection with source data)
                -1. if the index for a Tiler is not initialized (tile has not been calculated)
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
    def write_tiler(self, tile_generator: 'Tiler'):
        pass

    @abc.abstractmethod
    def read_tiler(self, tile: morecantile.Tile) -> 'Tiler' or None:
        """ Read tile_factory file from disk.

        Args:
            tile: ZXY Tile location

        Returns: The requested `Tiler`. `None` if it does not exist.

        """
        pass

    @abc.abstractmethod
    def check_tiler_exists(self, tile: morecantile.Tile) -> bool:
        pass
