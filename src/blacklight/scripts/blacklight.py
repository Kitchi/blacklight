#! /usr/bin/env python3

# This is the command line script. The importable libraries are within src/

import argparse

from blacklight import io, plot


def main():
    """
    UV plane visualization for radio interferometric data.
    """

    parser = argparse.ArgumentParser(
        description="UV plane visualization for radio interferometric data."
    )
    parser.add_argument("MS", type=str, help="Measurement Set file to visualize.")

    args = parser.parse_args()

    # Read the MS file
    ms = args.MS
    # Convert the MS to a polars dataframe
    df, pqname = io.ms_to_df(ms, persist=True, overwrite=False)
    plot.plot_uv_basic(df, xcol="U", ycol="V", zcol="AMP")
