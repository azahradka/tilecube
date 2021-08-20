import logging

import morecantile
import numpy as np
import pyproj
import scipy
import xesmf as xe
from morecantile import Tile, BoundingBox
from shapely.geometry import Polygon
from xarray import DataArray

from storage import PyramidStorage


class PyramidTile:
    def __init__(self, tile: Tile, weights, bounds, shape_in, shape_out):
        self.tile = tile
        self.weights = weights
        self.bounds = bounds
        self.shape_in = shape_in
        self.shape_out = shape_out

    def transform(self, data: DataArray):
        data_out = xe.smm.apply_weights(self.weights, data.values, self.shape_in, self.shape_out)
        return DataArray(data_out, dims=('y', 'x'))

    @property
    def xmin(self):
        return self.bounds[0]

    @property
    def ymin(self):
        return self.bounds[1]

    @property
    def xmax(self):
        return self.bounds[2]

    @property
    def ymax(self):
        return self.bounds[3]


class Pyramid:
    """ A pyramid of the the weight matrices used to transform the source data to a particular web map tile.

    Also stores information about the source grid and destination tile set. The `Pyramid` is what needs to be persisted
    to perform dynamic regridding of data matching the source grid.

    """

    # 256x256 is the standard web map tile size (pixels). Could make adjustable in the future.
    TILESIZE = 256

    def __init__(
            self,
            x: DataArray,
            y: DataArray,
            proj: pyproj.CRS = None,
            storage: PyramidStorage = None):
        """

        Args:
            x: `xarray.DataArray` of the X coordinates of the source grid. Can be one dimensional (rectilinear) or two
                dimensional (curvilinear).
            y: `xarray.DataArray` of the Y (vertical) coordinates of the source grid
            proj: Coordinate Reference System (CRS) of the source grid. Leave None to use Lat/Lon (EPSG:3857)
            storage: `PyramidStorage` object to use to persist the regridding weights and metadata. Leave None to
                avoid persistence and operate on-the-fly.
        """
        self.src_x = x
        self.src_y = y
        self.storage = storage
        if proj is None:
            self.src_proj = pyproj.CRS.from_epsg(3857)
        else:
            self.src_proj = proj

        # Tiles are in Web Mercator projection (EPSG:3857)
        self.tile_proj = pyproj.CRS.from_epsg(3857)
        self.tms = morecantile.tms.get('WebMercatorQuad')
        # We use WGS84 Lat/Lon (EPSG:4326) as an intermediary
        self.lonlat_proj = pyproj.CRS.from_epsg(4326)
        self.tile_lonlat_transformer = pyproj.Transformer.from_crs(
            self.tile_proj, self.lonlat_proj, always_xy=True)
        # If the source data is already in lat/lon, we can skip this reprojection
        if self.src_proj.equals(self.lonlat_proj):
            self.src_lonlat_tranformer = None
        else:
            self.src_lonlat_tranformer = pyproj.Transformer.from_crs(
                self.src_proj, self.lonlat_proj, always_xy=True)
        self.tile_src_transformer = pyproj.Transformer.from_crs(
            self.tile_proj, self.src_proj, always_xy=True)

        # We need the bounds of the data to determine if a given tile intersects the data.
        # This intersection is done in the source grid projection so the resolution of the source grid bounding poly
        # doesn't really matter.
        # TODO: does this work well when source is in a rectilinear grid?
        self.src_bounds = BoundingBox(
            self.src_x.min(),
            self.src_y.min(),
            self.src_x.max(),
            self.src_y.max())
        self.src_bounding_poly = self._get_bounding_poly(
            self.src_bounds.left,
            self.src_bounds.bottom,
            self.src_bounds.right,
            self.src_bounds.top)
        # Resolution of the source grid, in whatever units the source grid projection is in.
        self.src_resolution = (self.src_bounds.right - self.src_bounds.left) / len(self.src_x)

    def _get_bounding_poly(self, xmin, ymin, xmax, ymax, transform=None):
        """ Create a `shapely.Polygon` bounding box from bounding box corners for a grid of data.

        Optionally, reproject the bounding polygon. The bounding polygon is created with 256 vertices along each axis.

        Args:
            xmin:
            ymin:
            xmax:
            ymax:
            transform:

        Returns:

        """
        # 1-D Vector from x/y min to max, with 256 vertices
        tile_xvector = np.linspace(xmin, xmax, num=self.TILESIZE)
        tile_yvector = np.linspace(ymin, ymax, num=self.TILESIZE)
        # x coordinate vector is generated by concatenating:
        #   1. 256 coordinates of xmin ("left" side of rectangle)
        #   2. 256 coordinates evenly spaced from xmin to xmax ("top" of rectangle)
        #   3. 256 coordinates of xmax("right" side of rectangle)
        #   4. 256 coordinates evenly spaced from xmax to xmin ("bottom" of rectangle)
        x = np.concatenate([
            np.array([xmin] * self.TILESIZE),
            tile_xvector,
            np.array([xmax] * self.TILESIZE),
            tile_xvector[::-1]])
        # y coordinate vector is generated by concatenating:
        #   1. 256 coordinates evenly spaced from ymin to ymax ("left" side of rectangle)
        #   2. 256 coordinates of ymax ("top" of rectangle)
        #   3. 256 coordinates evenly spaced from ymax to ymin ("right" side of rectangle)
        #   4. 256 coordinates of ymin ("bottom" of rectangle)
        y = np.concatenate([
            tile_yvector,
            np.array([ymax] * self.TILESIZE),
            tile_yvector[::-1],
            np.array([ymin] * self.TILESIZE)])
        # Reproject the polygon coordinates if necessary
        if transform is not None:
            x, y = transform(x, y)
        # Create `shapely.Polygon` object from the coordinate vectors
        polygon = Polygon(zip(x, y))
        return polygon

    def _calculate_pyramid_tile(self, tile: Tile, method) -> PyramidTile or None:

        # Check to see if we already have already used an interpolation method for this zoom level.
        # If so make sure we are being consistent and using the same one
        existing_method = self.storage.read_method(tile)
        if existing_method is not None and existing_method != method:
            raise ValueError(f'Cannot create pyramid tile using interpolation method {method}'
                             f'when existing tiles at zoom level {tile.z} have been created'
                             f'using interpolation method {existing_method}.')

        # Create a bounding box for the web map tile in the source grid projection
        tile_bounds = self.tms.xy_bounds(tile)
        tile_bounding_poly = self._get_bounding_poly(
            tile_bounds.left,
            tile_bounds.bottom,
            tile_bounds.right,
            tile_bounds.top,
            self.tile_src_transformer.transform)
        clipped_data_bounding_poly = self.src_bounding_poly.intersection(tile_bounding_poly)
        if clipped_data_bounding_poly.is_empty:
            return None
        xmin, ymin, xmax, ymax = clipped_data_bounding_poly.bounds
        x = self.src_x.sel(x=slice(xmin - self.src_resolution, xmax + self.src_resolution))
        y = self.src_y.sel(y=slice(ymin - self.src_resolution, ymax + self.src_resolution))

        # Create grid of source coordinates in lat/long
        src_xgrid, src_ygrid = np.meshgrid(x, y)
        # We need to have some data in the source grids
        if 0 in src_xgrid.shape:
            return None
        # If the source data is already in lat/lon, we can skip this reprojection
        if self.src_lonlat_tranformer is None:
            src_longrid, src_latgrid = src_xgrid, src_ygrid
        else:
            src_longrid, src_latgrid = self.src_lonlat_tranformer.transform(src_xgrid, src_ygrid)

        # Create grid of tile coordinates in lat/long
        tile_bounds = self.tms.xy_bounds(tile)
        half_grid_width = ((tile_bounds.top - tile_bounds.bottom) / self.TILESIZE) / 2
        tile_yvector = np.linspace(
            tile_bounds.bottom + half_grid_width, tile_bounds.top - half_grid_width,
            num=self.TILESIZE)
        tile_xvector = np.linspace(
            tile_bounds.left + half_grid_width, tile_bounds.right - half_grid_width,
            num=self.TILESIZE)
        tile_xgrid, tile_ygrid = np.meshgrid(tile_xvector, tile_yvector)
        tile_longrid, tile_latgrid = self.tile_lonlat_transformer.transform(tile_xgrid, tile_ygrid)

        # Generate regridder from source to tile grids
        # with self.__suppress_stdout():
        log = logging.getLogger(__name__)
        print('Calculating regridder')
        regridder = xe.Regridder({'lon': src_longrid, 'lat': src_latgrid},
                                 {'lon': tile_longrid, 'lat': tile_latgrid}, method)
        print('Done.')

        X = regridder.weights
        M = scipy.sparse.csr_matrix(X)
        num_nonzeros = np.diff(M.indptr)
        M[num_nonzeros == 0, 0] = np.NaN
        regridder.weights = scipy.sparse.coo_matrix(M)
        bounds = (x.min(), y.min(), x.max(), y.max())
        pyramid_tile = PyramidTile(tile, regridder.weights, bounds, regridder.shape_in, regridder.shape_out)
        return pyramid_tile

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