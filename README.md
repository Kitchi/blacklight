# Blacklight

UV plane visualization for radio interferometric data. Converts CASA Measurement Sets into interactive Datashader-rasterized plots served via Panel, with Dask for out-of-core reads.

## Installation

```bash
# With pixi (recommended)
pixi install
pixi run pip install -e .

# Or plain pip
pip install -e .
```

## CLI Usage

```bash
blacklight <path-to-ms>                    # default settings
blacklight <path-to-ms> -n 8              # 8 parallel workers
blacklight <path-to-ms> --overwrite       # regenerate parquet cache
blacklight <path-to-ms> --port 5006       # specific Panel server port
blacklight <path-to-ms> --no-show         # start server without opening browser
blacklight <path-to-ms> --save-plot out.html  # static export (HTML or PNG), no server
```

## Python / Jupyter

```python
import dask.dataframe as dd
from blacklight import create_uv_plot, build_app

ddf = dd.read_parquet("my_data.ms.pq")

# Quick inline plot (renders in Jupyter via HoloViews)
create_uv_plot(ddf)

# Full interactive widget app
build_app(ddf).servable()
```

## Architecture

```
Measurement Set
  → io.ms_to_parquet()        # parallel CASA reads, scalar pre-compute, flag masking
  → Partitioned Parquet       # part.*.parquet files with flat scalars
  → dask.dataframe            # lazy out-of-core reads
  → plot.create_uv_plot()     # Datashader rasterization via HoloViews
  → app.build_app()           # Panel interactive app with sidebar widgets
```

Sidebar widgets: X/Y axis selectors, color aggregation, colormap, log scale toggle, field filter (multi-select).

## Dependencies

- `casatools`, `casatasks`, `casadata` (>= 6.7)
- `numpy < 2.0` (CASA compatibility)
- `pyarrow`, `dask[dataframe]`
- `datashader`, `holoviews`, `panel`
- `colorcet`
