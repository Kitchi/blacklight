"""
Plotting module — Datashader rasterization via HoloViews
"""

import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import dynspread, rasterize, spread

hv.extension("bokeh")

# Curated colormap choices (colorcet names → display labels)
COLORMAPS = {
    # Sequential
    "kbc": "Blue (kbc)",
    "fire": "Fire",
    "bgy": "Blue-Green-Yellow",
    "bgyw": "Blue-Green-Yellow-White",
    "bmw": "Blue-Magenta-White",
    "bmy": "Blue-Magenta-Yellow",
    "kgy": "Green-Yellow",
    "gray": "Grayscale",
    "dimgray": "Dark Grayscale",
    "kb": "Blue (linear)",
    "kr": "Red (linear)",
    "kg": "Green (linear)",
    # Diverging
    "coolwarm": "Cool-Warm",
    "bkr": "Blue-Black-Red",
    "bky": "Blue-Black-Yellow",
    "gwv": "Green-White-Violet",
    "CET_D1": "Diverging Blue-Red",
    # Rainbow / cyclic
    "rainbow4": "Rainbow",
    "isolum": "Isoluminant Rainbow",
    "colorwheel": "Color Wheel (cyclic)",
}

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
    cmap="kbc",
    logz=False,
    logx=False,
    logy=False,
    responsive=True,
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
    cmap : str
        Colormap name (colorcet key).
    logz : bool
        If True, use logarithmic color scale.
    logx : bool
        If True, use logarithmic x axis scale.
    logy : bool
        If True, use logarithmic y axis scale.
    responsive : bool
        If True, plot fills available space. Set False for static export.
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
    if logx or logy:
        rasterized = spread(rasterized, px=2)
    else:
        rasterized = dynspread(rasterized, max_px=4, threshold=0.5)

    xlabel = AXIS_COLUMNS.get(xcol, xcol)
    ylabel = AXIS_COLUMNS.get(ycol, ycol)
    clabel = COLOR_COLUMNS.get(agg_col, agg_col)

    rasterized = rasterized.opts(
        responsive=responsive,
        colorbar=True,
        clabel=clabel,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        logz=logz,
        logx=logx,
        logy=logy,
        tools=["hover"],
    )

    return rasterized
