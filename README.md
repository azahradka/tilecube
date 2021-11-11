# tilecube

Dynamic web map tile generation for data cubes

## Description

### Web Map Tiles

Most modern web maps display raster or image data using some derivation of the
[Slippy Map](https://wiki.openstreetmap.org/wiki/Slippy_Map) convention. In Slippy Map style web maps, the globe is
divided into a set of tiles at different zoom levels. At zoom level zero, one tile covers the globe, and each subsequent
zoom level divides previous tile into four. Traditionally, a more or less static, two dimensional raster dataset is
reprojected in the web-mercator projection and then preprocessed into tiles for each zoom level (up to a maximum zoom
level reflective of the resolution of the source raster). Preparing this tile cache can be a very computationally
expensive process resulting in a very large dataset which must be stored. Alternatively, tile generation can be done
on-the-fly but results in significantly slower map loading.

### Data Cubes

"Data cubes" are multidimensional ("n-D") data arrays (a.k.a. "tensors") which can be thought of as a stack of 2D raster
images. A data cube has a consistent grid in space (x and y) and one or more additional dimensions - commonly things
like time, elevation/altitude, and data variable. Data cubes can be stored as sets of 2D rasters (e.g. geotiffs) or in
n-D data array formats such as NetCDF.

### Tilecube

Pre-computing a tile cache to serve each permutation of data cube dimensions (besides x and y) can quickly become
extremely expensive. The problem grows exponentially when one additionally considers the possible calculations which can
be done over a data cube's dimensions, for example:

- differencing surface temperatures across two dates in time,
- summing precipitation a period of time, or
- performing calculations across multi-spectral satellite imagery bands

The Tilecube library is built upon the realization that the process of reprojecting a dataset and regridding from the
source data grid to a given web map
tile [is a linear transformation](https://xesmf.readthedocs.io/en/latest/notebooks/Reuse_regridder.html#Why-applying-regridding-is-so-fast?)
(in the case of most regridding algorithms). The transformation matrix to generate a given web map tile depends only on
the input grid and, once calculated, can be applied to any input data. This means that for any data cube, we can
pre-compute a single tile cache of transformation matrices, which is similar effort to calculating a tile cache for a
single 2D raster extracted from the data cube. A simple (and fast) matrix multiplication will then generate a tile for
any input data, be it selected from a dimension of the data cube or calculated on-the-fly. 
