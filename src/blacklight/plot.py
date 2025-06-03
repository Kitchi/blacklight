"""
Plotting module
"""

from bokeh.plotting import figure, show


def plot_uv_basic(uvdf, xcol: str = "U", ycol: str = "V", zcol: str = "AMP") -> None:
    """
    Basic UV plotting functionality, colorized by zcol

    TODO : Add signature
    """

    plot = figure(width=300, height=300, output_backend="webgl")
    plot.scatter(x=xcol, y=ycol, size=zcol, color="red", source=uvdf)
    show(plot)
