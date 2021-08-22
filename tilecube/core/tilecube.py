import morecantile
import numpy as np
import pyproj
import scipy
import xesmf as xe
from morecantile import Tile, BoundingBox
from shapely.geometry import Polygon
from xarray import DataArray

from tilecube.core.pyramid import PyramidTile
from tilecube.storage import TileCubeStorage


class TileCube:

    def __init__(self, storage: TileCubeStorage):
        self.storage = storage

    """
                storage: `PyramidStorage` object to use to persist the regridding weights and metadata. Leave None to
                avoid persistence and operate on-the-fly.

    """

    def get_pyramid_tile(self, tile, method):
        """ Read or calculate a PyramidTile.

        * If the pyramid tile exists in storage it is read out and returned.
        * If the pyramid tile does not exist in storage but can be calculated, it is calculated and returned.
        * If the pyramid tile cannot be calculated because there is no overlap between the data and tile,
            then None is returned.

        Args:
            tile:
            method:

        Returns: the requested PyramidTile. None if there is no overlap between the data and `tile`.

        """
        # Check to see if we already have already used an interpolation method for this zoom level.
        # If so make sure we are being consistent and using the same one
        existing_method = self.storage.read_method(tile)
        if existing_method is not None and existing_method != method:
            raise ValueError(f'Cannot create pyramid tile using interpolation method {method}'
                             f'when existing tiles at zoom level {tile.z} have been created'
                             f'using interpolation method {existing_method}.')

        if self.storage.read_parent_index(tile) is False:
            return None
        tile_index = self.storage.read_index(tile)
        if tile_index is False:
            return None
        elif tile_index is True:
            return self.storage.read_pyramid_tile(tile)
        else:
            return self._calculate_pyramid_tile(tile, method)

    def write_pyramid_tile(self, tile, method, overwrite=False):

        # Return the existing tile if one exists and overwrite is False
        if (not overwrite) and self.storage.check_pyramid_tile_exists(tile):
            return self.get_pyramid_tile(tile, method)

        # Get the pyramid
        pyramid_tile = self.get_pyramid_tile(tile, method)
        if pyramid_tile is None:
            self.storage.write_index(tile, False)
            return None
        self.storage.write_index(tile, True)
        self.storage.write_pyramid_tile(pyramid_tile)
        return pyramid_tile