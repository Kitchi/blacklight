"""
Plotting module
"""

import numpy as np
from bokeh.plotting import figure, show


# =============================================================================
# Derived quantities
# =============================================================================


def compute_mean_amplitude(df):
    """Compute mean amplitude per row from nested DATA_REAL/DATA_IMAG."""
    def row_amp(real, imag):
        return np.sqrt(np.mean(np.array(real) ** 2 + np.array(imag) ** 2))

    df["AMP"] = [row_amp(r, i) for r, i in zip(df["DATA_REAL"], df["DATA_IMAG"])]
    return df


def compute_mean_phase(df):
    """Compute mean phase (radians) per row from nested DATA_REAL/DATA_IMAG."""
    def row_phase(real, imag):
        return np.mean(np.arctan2(np.array(imag), np.array(real)))

    df["PHASE"] = [row_phase(r, i) for r, i in zip(df["DATA_REAL"], df["DATA_IMAG"])]
    return df


def compute_uvdist(df):
    """Compute UV distance (sqrt(U^2 + V^2)) per row."""
    df["UVDIST"] = np.sqrt(df["U"] ** 2 + df["V"] ** 2)
    return df


def add_derived_columns(df, amplitude=True, phase=False, uvdist=True):
    """
    Add commonly used derived columns to dataframe.

    Parameters
    ----------
    df : DataFrame
        Must have U, V columns. For amplitude/phase, needs DATA_REAL/DATA_IMAG.
    amplitude : bool
        Compute mean amplitude per row.
    phase : bool
        Compute mean phase per row.
    uvdist : bool
        Compute UV distance.

    Returns
    -------
    DataFrame
        Input dataframe with added columns.
    """
    if amplitude and "DATA_REAL" in df.columns:
        compute_mean_amplitude(df)
    if phase and "DATA_REAL" in df.columns:
        compute_mean_phase(df)
    if uvdist:
        compute_uvdist(df)
    return df


# =============================================================================
# Plotting
# =============================================================================


def plot_uv_basic(uvdf, xcol: str = "U", ycol: str = "V", zcol: str = "AMP") -> None:
    """
    Basic UV plotting functionality, colorized by zcol

    TODO : Add signature
    """

    plot = figure(width=300, height=300, output_backend="webgl")
    plot.scatter(x=xcol, y=ycol, size=2, color="red", source=uvdf)
    show(plot)
