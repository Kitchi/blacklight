#! /usr/bin/env python3

# This is the command line script. The importable libraries are within src/

import argparse

import dask.dataframe as dd

from blacklight import io
from blacklight.app import build_app


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
    parser.add_argument(
        "--port", type=int, default=0, help="Port for Panel server (0 = auto)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Start server without opening browser",
    )

    args = parser.parse_args()

    # Convert MS to partitioned parquet directory
    pqpath = io.ms_to_parquet(
        args.MS, nworkers=args.nworkers, overwrite=args.overwrite
    )

    # Lazy Dask DataFrame from partitioned parquet
    ddf = dd.read_parquet(pqpath)

    # Build and launch Panel application
    app = build_app(ddf)
    app.show(port=args.port, open=not args.no_show)
