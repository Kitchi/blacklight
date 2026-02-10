"""
Blacklight — UV plane visualization for radio interferometric data.
"""

import dask.dataframe as dd

from blacklight.app import build_app
from blacklight.io import get_ms_metadata, ms_to_parquet
from blacklight.plot import create_uv_plot

__all__ = ["ms_to_parquet", "get_ms_metadata", "create_uv_plot", "build_app", "view"]


def view(ms, output_pq=None, nworkers=None, overwrite=False, max_mem=None, title="Blacklight"):
    """
    Convert a Measurement Set to parquet (if needed) and return an
    interactive Panel application.

    Parameters
    ----------
    ms : str
        Path to a CASA Measurement Set.
    output_pq : str, optional
        Output parquet directory path. Defaults to ``<ms>.pq``.
    nworkers : int, optional
        Number of parallel workers for MS → parquet conversion.
    overwrite : bool
        If True, regenerate the parquet cache even if it exists.
    max_mem : float, optional
        Maximum total RAM budget in GB for parallel workers.
        Defaults to total system RAM.
    title : str
        Application title.

    Returns
    -------
    pn.Template
        Panel app. Call ``.servable()`` in a notebook or ``.show()``
        to launch a server.
    """
    pqpath = ms_to_parquet(ms, output_pq=output_pq, nworkers=nworkers, overwrite=overwrite, max_mem=max_mem)
    ddf = dd.read_parquet(pqpath)
    return build_app(ddf, title=title)
