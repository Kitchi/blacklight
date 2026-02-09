#! /usr/bin/env python3

# This is the command line script. The importable libraries are within src/

import argparse

import pyarrow.parquet as pq

from blacklight import io, plot


def main():
    """
    UV plane visualization for radio interferometric data.
    """

    parser = argparse.ArgumentParser(
        description="UV plane visualization for radio interferometric data."
    )
    parser.add_argument("MS", type=str, help="Measurement Set file to visualize.")
    parser.add_argument(
        "-n", "--nworkers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing parquet cache"
    )

    args = parser.parse_args()

    # Convert MS to parquet (parallel read)
    pqpath = io.ms_to_parquet(
        args.MS, nworkers=args.nworkers, overwrite=args.overwrite
    )

    # Read parquet for plotting
    table = pq.read_table(pqpath, columns=["U", "V", "DATA_REAL", "DATA_IMAG"])
    df = table.to_pandas()

    # Add derived columns, then drop the heavy nested arrays
    plot.add_derived_columns(df, amplitude=True, uvdist=True)
    df.drop(columns=["DATA_REAL", "DATA_IMAG"], inplace=True)

    plot.plot_uv_basic(df, xcol="U", ycol="V", zcol="AMP")
