from typing import List, Tuple

from morecantile import Tile

from tilecube.core import Tiler, TilerFactory


def calculate_tilers_distributed(
        tiler_factory,
        tiles: List[Tile],
        method,
        dask_client) -> List[Tuple[Tile, Tiler or None]]:
    tile_tuples = [(t.x, t.y, t.z) for t in tiles]
    tiler_factory_dict = tiler_factory.to_dict()
    future = dask_client.submit(
        dask_worker_calculate_tilers,
        tiles=tile_tuples,
        tiler_factory_dict=tiler_factory_dict,
        method=method)
    results = future.result()
    tilers = []
    for tile_index, tg_dict in results:
        tile = Tile(*tile_index)
        if tg_dict is not None:
            tg = Tiler.from_dict(tg_dict)
        else:
            tg = None
        tilers.append((tile, tg))
    return tilers


def dask_worker_calculate_tilers(
        tiles: List[Tuple[int, int, int]],
        tiler_factory_dict: dict,
        method: str) -> List[Tuple[Tuple[int, int, int], dict or None]]:
    tiler_factory = TilerFactory.from_dict(tiler_factory_dict)
    tilers = []
    for (x, y, z) in tiles:
        tile = Tile(x, y, z)
        tiler = tiler_factory.generate_tiler(tile, method)
        if tiler is not None:
            tiler_dict = tiler.to_dict()
        else:
            tiler_dict = None
        tilers.append(((x, y, z), tiler_dict))
    return tilers
