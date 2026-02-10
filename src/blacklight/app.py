"""
Panel application for interactive UV visualization
"""

import panel as pn

from blacklight.plot import AXIS_COLUMNS, COLORMAPS, COLOR_COLUMNS, create_uv_plot

pn.extension("bokeh")


def build_app(ddf, title="Blacklight") -> pn.Template:
    """
    Build an interactive Panel application for UV visualization.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Lazy Dask DataFrame from partitioned parquet.
    title : str
        Application title.

    Returns
    -------
    pn.Template
        Panel template ready for .show() or .servable().
    """
    axis_options = list(AXIS_COLUMNS.keys())
    color_options = list(COLOR_COLUMNS.keys())
    cmap_options = list(COLORMAPS.keys())

    # Populate field choices from the dataframe
    if "FIELD" in ddf.columns:
        field_options = sorted(ddf["FIELD"].unique().compute().tolist())
    else:
        field_options = []

    x_select = pn.widgets.Select(
        name="X Axis", options=axis_options, value="U"
    )
    y_select = pn.widgets.Select(
        name="Y Axis", options=axis_options, value="V"
    )
    color_select = pn.widgets.Select(
        name="Color", options=color_options, value="count"
    )
    cmap_select = pn.widgets.Select(
        name="Colormap", options=cmap_options, value="kbc"
    )
    logz_toggle = pn.widgets.Toggle(name="Log Scale", value=False)
    field_select = pn.widgets.MultiChoice(
        name="Field", options=field_options, value=[]
    )

    @pn.depends(x_select, y_select, color_select, cmap_select, logz_toggle, field_select)
    def _plot(xcol, ycol, agg_col, cmap, logz, fields):
        filtered = ddf
        if fields:
            filtered = ddf[ddf["FIELD"].isin(fields)]
        return create_uv_plot(
            filtered, xcol=xcol, ycol=ycol, agg_col=agg_col, cmap=cmap, logz=logz,
        )

    sidebar = [x_select, y_select, color_select, cmap_select, logz_toggle]
    if field_options:
        sidebar.append(field_select)

    template = pn.template.FastListTemplate(
        title=title,
        sidebar=sidebar,
        main=[pn.panel(_plot, sizing_mode="stretch_both")],
    )

    return template
