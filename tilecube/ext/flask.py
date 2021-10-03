import io
import typing as t
from functools import wraps

import flask
import matplotlib.colors
import matplotlib.pyplot as plt
import morecantile
import numpy as np
from PIL import Image

import tilecube


def tilemap(
    tilecube: tilecube.TileCube,
    method: str,
    cmap: t.Union[matplotlib.colors.Colormap, str] = None,
    norm: t.Callable = None,
) -> t.Callable:
    """

    :param tilecube:
    :param method:
    :param cmap:
    :param norm:
    :return:
    """
    def decorator(func: t.Callable) -> t.Callable:
        @wraps(func)
        def function_wrapper(z, y, x, *args, **kwargs) -> io.BytesIO:
            tile = morecantile.Tile(x, y, z)
            tiler = tilecube.tiler_factory.generate_tiler(tile, method)  # TODO variable methods
            if tiler is None:
                arr = np.empty((tilecube.tiler_factory.TILESIZE, tilecube.tiler_factory.TILESIZE))
                arr[:] = np.nan
            else:
                data = func((tiler.ymin, tiler.ymax, tiler.xmin, tiler.xmax), *args, **kwargs)
                if type(data) != np.ndarray and type(data) is not None:
                    raise(ValueError(f'The wrapped function returned {type(data)} instead of np.ndarray or None.'))
                arr = tiler.regrid(data)
                arr = np.flip(arr, axis=0)
            mask = np.isnan(arr)
            arr = np.ma.array(arr, mask=mask)
            if norm is not None:
                arr = norm(arr)
            if cmap is None:
                colormap = plt.get_cmap('viridis')
            elif type(cmap) == str:
                colormap = plt.get_cmap(cmap)
            else:
                colormap = cmap
            arr = colormap(arr)
            im = Image.fromarray(np.uint8(arr * 255))
            with io.BytesIO() as buffer:
                im.save(buffer, 'PNG')
                return flask.send_file(io.BytesIO(buffer.getvalue()), mimetype='image/png')
        return function_wrapper
    return decorator
