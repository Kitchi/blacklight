"""
Panel application for interactive UV visualization
"""

import colorcet as cc
import panel as pn

from blacklight.plot import AXIS_COLUMNS, COLORMAPS, COLOR_COLUMNS, create_uv_plot

pn.extension("bokeh")


def _cmap_swatch(cmap_name):
    """Return an HTML pane showing a colormap gradient preview."""
    try:
        palette = cc.palette[cmap_name]
    except KeyError:
        return pn.pane.HTML("")
    n = min(len(palette), 30)
    step = max(1, len(palette) // n)
    colors = [palette[i * step] for i in range(n)]
    stops = ", ".join(colors)
    return pn.pane.HTML(
        f'<div style="height:14px;width:100%;'
        f"background:linear-gradient(to right, {stops});"
        f'border-radius:3px;margin-top:2px;"></div>',
        sizing_mode="stretch_width",
    )


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
        field_options = sorted(ddf["FIELD"].unique().tolist())
    else:
        field_options = []

    x_select = pn.widgets.Select(
        name="X Axis", options=axis_options, value="U"
    )
    y_select = pn.widgets.Select(
        name="Y Axis", options=axis_options, value="V"
    )
    logx_toggle = pn.widgets.Toggle(name="Log X", value=False, button_type="danger")
    logy_toggle = pn.widgets.Toggle(name="Log Y", value=False, button_type="danger")
    color_select = pn.widgets.Select(
        name="Color", options=color_options, value="count"
    )
    cmap_select = pn.widgets.Select(
        name="Colormap", options=cmap_options, value="kbc"
    )
    # Reactive colormap preview (cheap — updates without button)
    cmap_preview = pn.panel(pn.bind(_cmap_swatch, cmap_select))
    logz_toggle = pn.widgets.Toggle(name="Log Color", value=False, button_type="danger")

    def _style_toggle(toggle):
        toggle.button_type = "primary" if toggle.value else "danger"
        def _on_change(event):
            toggle.button_type = "primary" if event.new else "danger"
        toggle.param.watch(_on_change, "value")

    for t in (logx_toggle, logy_toggle, logz_toggle):
        _style_toggle(t)

    field_select = pn.widgets.MultiChoice(
        name="Field", options=field_options, value=[]
    )
    update_btn = pn.widgets.Button(name="Update Plot", button_type="primary")

    plot_pane = pn.pane.HoloViews(None, sizing_mode="stretch_both")

    def _update_plot(event=None):
        filtered = ddf
        if field_select.value:
            filtered = ddf[ddf["FIELD"].isin(field_select.value)]
        plot_pane.object = create_uv_plot(
            filtered,
            xcol=x_select.value,
            ycol=y_select.value,
            agg_col=color_select.value,
            cmap=cmap_select.value,
            logz=logz_toggle.value,
            logx=logx_toggle.value,
            logy=logy_toggle.value,
        )

    update_btn.on_click(_update_plot)
    _update_plot()  # initial render

    sidebar = [
        x_select, y_select, logx_toggle, logy_toggle,
        color_select, cmap_select, cmap_preview, logz_toggle,
    ]
    if field_options:
        sidebar.append(field_select)
    sidebar.append(update_btn)

    template = pn.template.FastListTemplate(
        title=title,
        sidebar=sidebar,
        main=[plot_pane],
    )

    return template
