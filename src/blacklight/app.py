"""
Panel application for interactive UV visualization
"""

import panel as pn

from blacklight.plot import AXIS_COLUMNS, COLOR_COLUMNS, create_uv_plot

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

    x_select = pn.widgets.Select(
        name="X Axis", options=axis_options, value="U"
    )
    y_select = pn.widgets.Select(
        name="Y Axis", options=axis_options, value="V"
    )
    color_select = pn.widgets.Select(
        name="Color", options=color_options, value="count"
    )

    @pn.depends(x_select, y_select, color_select)
    def _plot(xcol, ycol, agg_col):
        return create_uv_plot(ddf, xcol=xcol, ycol=ycol, agg_col=agg_col)

    template = pn.template.FastListTemplate(
        title=title,
        sidebar=[x_select, y_select, color_select],
        main=[pn.panel(_plot, sizing_mode="stretch_both")],
    )

    return template
