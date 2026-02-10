#! /usr/bin/env python3

# This is the command line script. The importable libraries are within src/

import argparse

import dask.dataframe as dd
import holoviews as hv

from blacklight import io
from blacklight.app import build_app
from blacklight.plot import create_uv_plot


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
        "--output-pq", type=str, default=None, metavar="PATH",
        help="Output parquet directory path (default: <MS>.pq)",
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
    parser.add_argument(
        "--max-mem",
        type=float,
        default=None,
        metavar="GB",
        help="Maximum RAM budget in GB for parallel workers (default: system RAM)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a static plot to PATH (.html or .png) and exit",
    )

    args = parser.parse_args()

    # Convert MS to partitioned parquet directory
    max_mem_bytes = args.max_mem * 1024**3 if args.max_mem else None
    pqpath = io.ms_to_parquet(
        args.MS, output_pq=args.output_pq, nworkers=args.nworkers,
        overwrite=args.overwrite, max_mem=max_mem_bytes,
    )

    # Lazy Dask DataFrame from partitioned parquet
    ddf = dd.read_parquet(pqpath)

    if args.save_plot:
        # Static export — no server
        element = create_uv_plot(ddf, responsive=False)
        hv.save(element, args.save_plot)
        print(f"Plot saved to {args.save_plot}")
        return

    # Build and launch Panel application
    app = build_app(ddf)
    app.show(port=args.port, open=not args.no_show)
