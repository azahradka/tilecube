from typing import Tuple

import morecantile
import numpy as np
import pyproj
import scipy
import scipy.sparse
import xesmf as xe
import xesmf.smm
from morecantile import Tile
from shapely import geometry
from shapely.strtree import STRtree
import shapely.vectorized


class Tiler:
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
    def from_dict(cls, d: dict) -> 'Tiler':
        tiler = Tiler(**d)
        return tiler

    def regrid(self, data: np.array):
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


class TilerFactory:
    """ Generates `Tiler` objects for any zoom, x, y tile location based on the source data grid

    """

    # 256x256 is the standard web map tile size (pixels). Could make adjustable in the future.
    TILESIZE = 256

    def __init__(
            self,
            y: np.array,
            x: np.array,
            proj: pyproj.CRS = None):
        """

        A note on the order of the x and y axes: In geospatial applications, care must be given to the order in which
        the x and y axes are presented. Unlike in general cartesian applications, the y axis, commonly represented by
        degrees latitude, is often presented before the x axis
        (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#dimensions). Within the
        TileCube project, the axis-order for data and coordinates is y, then x. This applies both to the order in which the
        coordinate variables are passed, and the order in which the data and 2-d coordinate arrays are indexed.

        Args:
            x: array of the X coordinates of the source grid. Can be one dimensional for rectilinear grids or two
                dimensional for general curvilinear grids. Two dimensional grids must have axis order (y, x).
            y: array of the Y coordinates of the source grid. Can be one dimensional for rectilinear grids or two
                dimensional for general curvilinear grids. Two dimensional grids must have axis order (y, x).
            proj: Coordinate Reference System (CRS) of the source grid. Defaults to Lat/Lon (EPSG:3857)
        """
        self.src_y = y
        self.src_x = x
        self.ygrid, self.xgrid = self._input_grid_to_2d(y, x)
        # There is no widely followed convention for the order of the y-axis in geospatial array datasets
        # This corresponds to the 'spatial reference point' of the array being in the top left (y decreasing) or
        # bottom left (y increasing). The convention for exporting png tiles from numpy is to have the reference point
        # in the top left so we will make sure that convention is followed for all processing within the TileCube.
        self.y_flipped, self.ygrid = self._correct_flipped_y_orientation(self.ygrid)

        # Take the x and y coordinate grids and generate a table of the grid points:
        #   x-axis index, y-axis index, x-coordinate, y-coordinate
        # This table can then be used to filter the source grid points which fall within a given tile
        self.iyv, self.ixv, self.yv, self.xv = self._yxgrid_to_table(self.ygrid, self.xgrid)
        # Shapely geometries in the points table
        self.xyv_points = [geometry.Point(x, y) for x, y in zip(self.xv, self.yv)]
        # RTree spatial index of the points table
        self.xyv_points_index = STRtree(self.xyv_points)
        # Index of the RTree spatial index, used to lookup the points table index position based on RTree geometries
        #   (This is needed because the RTree returns the geometries themselves, rather than the index positions of the
        #   geometries)
        self.xyv_points_index_ids = dict((id(pt), i) for i, pt in enumerate(self.xyv_points))

        # Setup Projections
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
            if self.src_x[1].min() < -180 or self.src_x[1].max() > 180:
                raise ValueError('The supplied Longitude (x) values must be in the range of -180 <= x <= 180')
        else:
            self.src_lonlat_tranformer = pyproj.Transformer.from_crs(
                self.src_proj, self.lonlat_proj, always_xy=True)
        self.tile_src_transformer = pyproj.Transformer.from_crs(
            self.tile_proj, self.src_proj, always_xy=True)

    @staticmethod
    def _correct_flipped_y_orientation(ygrid):
        if ygrid[-1, 0] > ygrid[0, 0]:
            y_flipped = True
            corrected_ygrid = ygrid[::-1, :]
        else:
            y_flipped = False
            corrected_ygrid = ygrid
        return y_flipped, corrected_ygrid

    @staticmethod
    def _input_grid_to_2d(y, x):
        """ If x and y are 1-dimensional vectors, generate 2-d x and y coordinate grids """
        if y.ndim == 1 and x.ndim == 1:
            ygrid, xgrid = np.meshgrid(y, x, indexing='ij')
        elif y.ndim == 2 and x.ndim == 2 :
            if y.shape != x.shape:
                raise ValueError('x and y must have matching dimensions')
            ygrid = y
            xgrid = x
        else:
            raise ValueError('x and y must be one or two dimensional')
        return ygrid, xgrid

    @staticmethod
    def _yxgrid_to_table(y, x) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Convert input x, y grid into a table of index, coordinate rows

        Args:
            x: 2D array of x coordinates
            y: 2D array of y coordinates

        Returns: four numpy arrays of equal length, containing:
            1. x-axis index position
            2. y-axis index position
            3. x coordinate value
            4. y coordinate value

        """
        iy = np.repeat(np.arange(y.shape[0]), y.shape[1])
        ix = np.hstack([np.arange(x.shape[1])] * x.shape[0])
        yv = np.ravel(y)
        xv = np.ravel(x)
        return iy, ix, yv, xv

    @staticmethod
    def _bbox_to_poly(xmin, ymin, xmax, ymax, transform=None) -> geometry.Polygon:
        """ Create a `shapely.Polygon` bounding box from standard bounding box corners.

        Optionally, reproject the bounding polygon. The bounding polygon is created with 256 vertices along each axis.
        (256 because that is the number of grid points along an axis of a standard web map tile.

        Args:
            xmin: x-coordinate of the lower left corner of the bounding box
            ymin: y-coordinate of the lower left corner of the bounding box
            xmax: x-coordinate of the upper right corner of the bounding box
            ymax: y-coordinate of the upper right corner of the bounding box
            transform: optional transformation function to apply to the bounding box polygon coordinates
                This is generally a `pyproj.Transformer` transformation function.

        Returns: The (optionally reprojected) polygon bounding box

        """
        # Vectors from x/y min to max, with 256 vertices
        tile_xvector = np.linspace(xmin, xmax, num=TilerFactory.TILESIZE)
        tile_yvector = np.linspace(ymin, ymax, num=TilerFactory.TILESIZE)
        # x coordinate vector is generated by concatenating:
        #   1. 256 coordinates of xmin ("left" side of rectangle)
        #   2. 256 coordinates evenly spaced from xmin to xmax ("top" of rectangle)
        #   3. 256 coordinates of xmax("right" side of rectangle)
        #   4. 256 coordinates evenly spaced from xmax to xmin ("bottom" of rectangle)
        x = np.concatenate([
            np.array([xmin] * TilerFactory.TILESIZE),
            tile_xvector,
            np.array([xmax] * TilerFactory.TILESIZE),
            tile_xvector[::-1]])
        # y coordinate vector is generated by concatenating:
        #   1. 256 coordinates evenly spaced from ymin to ymax ("left" side of rectangle)
        #   2. 256 coordinates of ymax ("top" of rectangle)
        #   3. 256 coordinates evenly spaced from ymax to ymin ("right" side of rectangle)
        #   4. 256 coordinates of ymin ("bottom" of rectangle)
        y = np.concatenate([
            tile_yvector,
            np.array([ymax] * TilerFactory.TILESIZE),
            tile_yvector[::-1],
            np.array([ymin] * TilerFactory.TILESIZE)])
        # Reproject the polygon coordinates if necessary
        if transform is not None:
            x, y = transform(x, y)
        # Create `shapely.Polygon` object from the coordinate vectors
        polygon = geometry.Polygon(zip(x, y))
        return polygon

    def to_dict(self) -> dict:
        proj = self.src_proj.to_proj4()
        return dict(
            y=self.src_y,
            x=self.src_x,
            proj=proj)

    @classmethod
    def from_dict(cls, d: dict) -> 'TilerFactory':
        proj = pyproj.CRS.from_proj4(d['proj'])
        return TilerFactory(d['y'], d['x'], proj)

    def generate_tiler(self, tile: Tile, method) -> Tiler or None:
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

        Returns: The generated Tiler for the requested xyz tile index
            None if there is no intersection of the source data with the given tile
            The Tiler includes the tranformation matrix to reproject and regrid the source dataset to the given
                web map tile.

        """

        # Create a bounding box for the web map tile, in the source grid projection
        tile_bounds = self.tile_matrix_set.xy_bounds(tile)
        tile_bounding_poly = self._bbox_to_poly(
            tile_bounds.left,
            tile_bounds.bottom,
            tile_bounds.right,
            tile_bounds.top,
            self.tile_src_transformer.transform)

        # Select grid of source coordinates which intersect the current tile and then reproject to lat/long
        # Include an additional grid point on each side because we need to have the first data point outside a
        # given tile domain for some regridding algorithms to work (e.g. "conservative").
        mask: np.ndarray = shapely.vectorized.contains(tile_bounding_poly, self.xv, self.yv)
        if sum(mask) > 0:
            # Select the min/max index values for the x and y coordinate axes.
            # If possible, take one additional index position below/above the min/max, respectively.
            # Note that ix_max and iy_max have an addition +1 to make the endpoint inclusive during numpy
            #   endpoint-exclusive slicing
            ix_min = max(np.min(self.ixv[mask]) - 1, np.min(self.ixv))
            ix_max = min(np.max(self.ixv[mask]) + 1, np.max(self.ixv)) + 1
            iy_min = max(np.min(self.iyv[mask]) - 1, np.min(self.iyv))
            iy_max = min(np.max(self.iyv[mask]) + 1, np.max(self.iyv)) + 1
            src_ygrid_subset = self.ygrid[iy_min:iy_max, ix_min:ix_max]
            src_xgrid_subset = self.xgrid[iy_min:iy_max, ix_min:ix_max]
        else:
            # If there are no intersecting points, this could be because:
            #   1. The tile bounds do not intersect the source data
            #   2. The tile bounds intersect the source data, but do not contain any of the grid points
            # First, find the nearest grid point to the tile boundary, then buffer one point out in each direction.
            # If this grid region contains the tile, then this is the relevant grid region for regridding.
            # If the grid region doesn't contain the tile, then the tile bounds are outside the source grid extent.
            # Note that ix_max and iy_max have an addition +1 to make the endpoint inclusive during numpy
            #   endpoint-exclusive slicing
            nearest_point = self.xyv_points_index.nearest(tile_bounding_poly.centroid)
            nearest_point_id = self.xyv_points_index_ids[id(nearest_point)]
            ix_min = max(self.ixv[nearest_point_id] - 1, np.min(self.ixv))
            ix_max = min(self.ixv[nearest_point_id] + 1, np.max(self.ixv)) + 1
            iy_min = max(self.iyv[nearest_point_id] - 1, np.min(self.iyv))
            iy_max = min(self.iyv[nearest_point_id] + 1, np.max(self.iyv)) + 1
            src_ygrid_subset = self.ygrid[iy_min:iy_max, ix_min:ix_max]
            src_xgrid_subset = self.xgrid[iy_min:iy_max, ix_min:ix_max]
            grid_subset_bbox = geometry.box(
                np.min(src_xgrid_subset),
                np.min(src_ygrid_subset),
                np.max(src_xgrid_subset),
                np.max(src_ygrid_subset)
            )
            if not grid_subset_bbox.intersects(tile_bounding_poly):
                return None

        # Take the portion of the source grid which is relevant for the current tile and reproject it to
        # lat/lon (WGS84) (if it's not already) because the regridding will be done in lat/lon.
        if self.src_lonlat_tranformer is None:
            src_longrid_subset, src_latgrid_subset = src_xgrid_subset, src_ygrid_subset
        else:
            src_longrid_subset, src_latgrid_subset = self.src_lonlat_tranformer.transform(src_xgrid_subset, src_ygrid_subset)

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
        regridder = xe.Regridder({'lon': src_longrid_subset, 'lat': src_latgrid_subset},
                                 {'lon': tile_longrid, 'lat': tile_latgrid}, method)
        X = regridder.weights
        M = scipy.sparse.csr_matrix(X)
        num_nonzeros = np.diff(M.indptr)
        M[num_nonzeros == 0, 0] = np.NaN
        regridder.weights = scipy.sparse.coo_matrix(M)
        bounds = (ix_min, iy_min, ix_max, iy_max)
        tiler = Tiler(
            tile,
            regridder.weights,
            bounds,
            regridder.shape_in,
            regridder.shape_out,
            self.y_flipped)
        return tiler
