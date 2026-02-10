"""
Plotting module — Datashader rasterization via HoloViews
"""

import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import rasterize

hv.extension("bokeh")

# Available axis choices
AXIS_COLUMNS = {
    "U": "U (wavelengths)",
    "V": "V (wavelengths)",
    "W": "W (wavelengths)",
    "TIME": "Time",
    "UVDIST": "UV Distance",
    "AMP": "Mean Amplitude",
    "PHASE": "Mean Phase (rad)",
}

# Columns that can be used for color aggregation
COLOR_COLUMNS = {
    "count": "Point Density",
    "AMP": "Mean Amplitude",
    "PHASE": "Mean Phase (rad)",
    "UVDIST": "UV Distance",
}

# Map color column names to datashader aggregators
AGGREGATORS = {
    "count": ds.count(),
    "AMP": ds.mean("AMP"),
    "PHASE": ds.mean("PHASE"),
    "UVDIST": ds.mean("UVDIST"),
}


def create_uv_plot(
    ddf,
    xcol="U",
    ycol="V",
    agg_col="count",
    width=800,
    height=800,
) -> hv.Element:
    """
    Create a rasterized scatter plot from a Dask DataFrame.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        DataFrame with columns matching xcol, ycol, and agg_col.
    xcol : str
        Column for the x axis.
    ycol : str
        Column for the y axis.
    agg_col : str
        Aggregation column. "count" for density, or a column name
        for ds.mean() aggregation.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.

    Returns
    -------
    hv.Element
        Rasterized HoloViews element.
    """
    points = hv.Points(ddf, kdims=[xcol, ycol])

    aggregator = AGGREGATORS.get(agg_col, ds.count())

    rasterized = rasterize(points, aggregator=aggregator, width=width, height=height)

    xlabel = AXIS_COLUMNS.get(xcol, xcol)
    ylabel = AXIS_COLUMNS.get(ycol, ycol)
    clabel = COLOR_COLUMNS.get(agg_col, agg_col)

    rasterized = rasterized.opts(
        width=width,
        height=height,
        colorbar=True,
        clabel=clabel,
        xlabel=xlabel,
        ylabel=ylabel,
        tools=["hover"],
        sizing_mode="stretch_both",
    )

    return rasterized
