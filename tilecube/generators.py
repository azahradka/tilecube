import pyproj
import xesmf as xe
import xesmf.smm
import morecantile
import numpy as np
import scipy
import scipy.sparse
from morecantile import Tile, BoundingBox
from shapely.geometry import Polygon


class TileGenerator:
    """ Transforms source data into a given web map grid

    """
    def __init__(self, tile: Tile, weights, bounds, shape_in, shape_out, y_flipped=False):
        self.tile = tile
        self.weights = weights
        self.bounds = bounds
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.y_flipped = y_flipped

    def to_dict(self) -> dict:
        d = self.__dict__
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TileGenerator':
        tg = TileGenerator(**d)
        return tg

    def transform(self, data: np.array):
        vals = data
        if self.y_flipped:
            vals = vals[::-1, :]
        data_out = xe.smm.apply_weights(self.weights, vals, self.shape_in, self.shape_out)
        return data_out

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


class PyramidGenerator:
    """ A pyramid of the the weight matrices used to transform the source data to a particular web map tile.

    Also stores information about the source grid and destination tile set. The `Pyramid` is what needs to be persisted
    to perform dynamic regridding of data matching the source grid.

    """

    # 256x256 is the standard web map tile size (pixels). Could make adjustable in the future.
    TILESIZE = 256

    def __init__(
            self,
            x: np.array,
            y: np.array,
            proj: pyproj.CRS = None):
        """

        Args:
            x: array of the X coordinates of the source grid. Must be one dimensional (rectilinear).
            y: array of the Y coordinates of the source grid. Must be one dimensional (rectilinear).
            proj: Coordinate Reference System (CRS) of the source grid. Leave None to use Lat/Lon (EPSG:3857)
        """
        if x.ndim != 1 or y.ndim != 1:
            raise NotImplementedError("Grid must currently be rectilinear (1-d x and y)")
        self.src_x = x
        # TODO comment
        if y[0] > y[-1]:
            self.y_flipped = True
            self.src_y = y[::-1]
        else:
            self.y_flipped = False
            self.src_y = y
        if proj is None:
            self.src_proj = pyproj.CRS.from_epsg(4326)
        else:
            self.src_proj = proj

        # Tiles are in Web Mercator projection (EPSG:3857)
        self.tile_proj = pyproj.CRS.from_epsg(3857)
        self.tile_matrix_set = morecantile.tms.get('WebMercatorQuad')
        # We use WGS84 Lat/Lon (EPSG:4326) as an intermediary
        self.lonlat_proj = pyproj.CRS.from_epsg(4326)
        self.tile_lonlat_transformer = pyproj.Transformer.from_crs(
            self.tile_proj, self.lonlat_proj, always_xy=True)
        # If the source data is already in lat/lon, we can skip this reprojection
        if self.src_proj.equals(self.lonlat_proj):
            self.src_lonlat_tranformer = None
            if self.src_x.min() < -180 or self.src_x.max() > 180:
                raise ValueError('The supplied Longitude (x) values must be in the range of -180 <= x <= 180')
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
            float(self.src_x.min()),
            float(self.src_y.min()),
            float(self.src_x.max()),
            float(self.src_y.max()))
        self.src_bounding_poly = self.get_bounding_poly(
            self.src_bounds.left,
            self.src_bounds.bottom,
            self.src_bounds.right,
            self.src_bounds.top)
        # Resolution of the source grid, in whatever units the source grid projection is in.
        self.src_resolution = (self.src_bounds.right - self.src_bounds.left) / len(self.src_x)

    def to_dict(self) -> dict:
        return dict(
            x=self.src_x,
            y=self.src_y,
            proj=self.src_proj.to_proj4())

    @classmethod
    def from_dict(cls, d: dict) -> 'PyramidGenerator':
        proj = pyproj.CRS.from_proj4(d['proj'])
        return PyramidGenerator(d['x'], d['y'], proj)

    @staticmethod
    def get_bounding_poly(xmin, ymin, xmax, ymax, transform=None):
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
        # Vectors from x/y min to max, with 256 vertices
        tile_xvector = np.linspace(xmin, xmax, num=PyramidGenerator.TILESIZE)
        tile_yvector = np.linspace(ymin, ymax, num=PyramidGenerator.TILESIZE)
        # x coordinate vector is generated by concatenating:
        #   1. 256 coordinates of xmin ("left" side of rectangle)
        #   2. 256 coordinates evenly spaced from xmin to xmax ("top" of rectangle)
        #   3. 256 coordinates of xmax("right" side of rectangle)
        #   4. 256 coordinates evenly spaced from xmax to xmin ("bottom" of rectangle)
        x = np.concatenate([
            np.array([xmin] * PyramidGenerator.TILESIZE),
            tile_xvector,
            np.array([xmax] * PyramidGenerator.TILESIZE),
            tile_xvector[::-1]])
        # y coordinate vector is generated by concatenating:
        #   1. 256 coordinates evenly spaced from ymin to ymax ("left" side of rectangle)
        #   2. 256 coordinates of ymax ("top" of rectangle)
        #   3. 256 coordinates evenly spaced from ymax to ymin ("right" side of rectangle)
        #   4. 256 coordinates of ymin ("bottom" of rectangle)
        y = np.concatenate([
            tile_yvector,
            np.array([ymax] * PyramidGenerator.TILESIZE),
            tile_yvector[::-1],
            np.array([ymin] * PyramidGenerator.TILESIZE)])
        # Reproject the polygon coordinates if necessary
        if transform is not None:
            x, y = transform(x, y)
        # Create `shapely.Polygon` object from the coordinate vectors
        polygon = Polygon(zip(x, y))
        return polygon

    def calculate_tile_generator(self, tile: Tile, method) -> TileGenerator or None:
        """

        Args:
            tile: The xyz web map tile to generate
            method: The xESMF regridding algorithm to use
                One of:
                    ["bilinear",
                    "conservative",
                    "nearest_s2d",
                    "nearest_d2s",
                    "patch"]
                See https://pangeo-xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html for details

        Returns: The generated PyramidTile for the requested xyz tile index
            None if there is no intersection of the source data with the given tile
            The PyramidTile includes the tranformation matrix to reproject and regrid the source dataset to the given
                web map tile.

        """

        # Create a bounding box for the web map tile, in the source grid projection
        tile_bounds = self.tile_matrix_set.xy_bounds(tile)
        tile_bounding_poly = self.get_bounding_poly(
            tile_bounds.left,
            tile_bounds.bottom,
            tile_bounds.right,
            tile_bounds.top,
            self.tile_src_transformer.transform)
        # Determine area of the source grid which intersects the current tile (still in source grid projection)
        clipped_data_bounding_poly = self.src_bounding_poly.intersection(tile_bounding_poly)
        # If no part of the source grid intersects the current tile then there is nothing else to calculate
        if clipped_data_bounding_poly.is_empty:
            return None

        # Create grid of source coordinates which intersect the current tile, in lat/long
        # Start by selecting the region of the source grid which is within the current tile bounds
        #   But, include an additional grid point on each side because we need to have the first data point outside a
        #   given tile domain for some regridding algorithms to work (e.g. "conservative").
        xmin, ymin, xmax, ymax = clipped_data_bounding_poly.bounds
        x = self.src_x[np.where(
            (self.src_x >= (xmin - self.src_resolution))
            & (self.src_x <= (xmax + self.src_resolution)))]
        y = self.src_y[np.where(
            (self.src_y >= (ymin - self.src_resolution))
            & (self.src_y <= (ymax + self.src_resolution)))]
        src_xgrid, src_ygrid = np.meshgrid(x, y)
        # Confirm we have data in the source grids
        if 0 in src_xgrid.shape:
            return None
        # Reproject the source data grid for the current tile into lat/lon
        # If the source data is already in lat/lon, we can skip this reprojection
        if self.src_lonlat_tranformer is None:
            src_longrid, src_latgrid = src_xgrid, src_ygrid
        else:
            src_longrid, src_latgrid = self.src_lonlat_tranformer.transform(src_xgrid, src_ygrid)

        # Create grid of tile coordinates in lat/long
        # Note that `half_grid_width` is compensated for because the tile bounds are reported to the exterior edge of
        #   the tile grid cells, while the grid coordinates are referenced to the center of each grid cell.
        tile_bounds = self.tile_matrix_set.xy_bounds(tile)
        half_grid_width = ((tile_bounds.top - tile_bounds.bottom) / self.TILESIZE) / 2
        tile_yvector = np.linspace(
            tile_bounds.bottom + half_grid_width, tile_bounds.top - half_grid_width,
            num=self.TILESIZE)
        tile_xvector = np.linspace(
            tile_bounds.left + half_grid_width, tile_bounds.right - half_grid_width,
            num=self.TILESIZE)
        tile_xgrid, tile_ygrid = np.meshgrid(tile_xvector, tile_yvector)
        tile_longrid, tile_latgrid = self.tile_lonlat_transformer.transform(tile_xgrid, tile_ygrid)

        # Generate `xesmf.Regridder` from source grid to tile grid
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
        pyramid_tile = TileGenerator(tile, regridder.weights, bounds, regridder.shape_in, regridder.shape_out, self.y_flipped)
        return pyramid_tile

