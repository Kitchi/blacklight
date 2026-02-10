"""
IO module
"""

import math
import multiprocessing
import os
import shutil
from multiprocessing import cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from casatools import msmetadata, table


# =============================================================================
# Metadata and chunk utilities
# =============================================================================


def get_ms_metadata(input_ms: str) -> dict:
    """
    Extract metadata from a Measurement Set.

    Parameters
    ----------
    input_ms : str
        Path to the measurement set.

    Returns
    -------
    dict
        Dictionary containing:
        - nrow: total number of rows
        - nspw: number of spectral windows
        - npol: number of polarizations
        - nchan: number of channels (per spw, dict)
        - time_range: (min, max) TIME values
        - has_corrected: whether CORRECTED_DATA column exists
        - has_data: whether DATA column exists
    """
    _tb = table()
    _msmd = msmetadata()

    _tb.open(input_ms)
    nrow = _tb.nrows()
    colnames = _tb.colnames()
    time_col = _tb.getcol("TIME")
    _tb.close()

    _msmd.open(input_ms)
    nspw = _msmd.nspw()
    npol = _msmd.ncorrforpol(0)  # assume consistent across pols
    nchan = {spw: _msmd.nchan(spw) for spw in range(nspw)}
    _msmd.close()

    return {
        "nrow": nrow,
        "nspw": nspw,
        "npol": npol,
        "nchan": nchan,
        "time_range": (time_col.min(), time_col.max()),
        "has_corrected": "CORRECTED_DATA" in colnames,
        "has_data": "DATA" in colnames,
    }


def compute_time_chunks(input_ms: str, nchunks: int = None) -> list[tuple[int, int]]:
    """
    Compute row ranges partitioned by time boundaries.

    Splits the MS into approximately nchunks chunks, respecting integration
    boundaries (won't split mid-integration).

    Parameters
    ----------
    input_ms : str
        Path to the measurement set.
    nchunks : int, optional
        Number of chunks to create. Defaults to cpu_count().

    Returns
    -------
    list[tuple[int, int]]
        List of (startrow, nrow) tuples.
    """
    if nchunks is None:
        nchunks = cpu_count()

    _tb = table()
    _tb.open(input_ms)
    time_col = _tb.getcol("TIME")
    nrow = _tb.nrows()
    _tb.close()

    # Find unique time values and their first occurrence indices
    unique_times, first_indices = np.unique(time_col, return_index=True)
    n_times = len(unique_times)

    if n_times <= nchunks:
        # More chunks than time steps - one chunk per time step
        chunks = []
        sorted_indices = np.argsort(first_indices)
        for i, idx in enumerate(sorted_indices):
            start = first_indices[idx]
            if i + 1 < len(sorted_indices):
                end = first_indices[sorted_indices[i + 1]]
            else:
                end = nrow
            chunks.append((int(start), int(end - start)))
        return chunks

    # Distribute time steps across chunks
    times_per_chunk = n_times // nchunks
    chunks = []
    sorted_indices = np.argsort(first_indices)

    for w in range(nchunks):
        start_time_idx = w * times_per_chunk
        if w == nchunks - 1:
            end_time_idx = n_times
        else:
            end_time_idx = (w + 1) * times_per_chunk

        startrow = first_indices[sorted_indices[start_time_idx]]
        if end_time_idx >= n_times:
            endrow = nrow
        else:
            endrow = first_indices[sorted_indices[end_time_idx]]

        chunks.append((int(startrow), int(endrow - startrow)))

    return chunks


# =============================================================================
# Parallel read worker and orchestrator
# =============================================================================


def _read_chunk(args: tuple) -> str:
    """
    Worker function: read a row range from MS, compute scalar columns,
    and write directly to a part file.

    Parameters
    ----------
    args : tuple
        (input_ms, startrow, nrow, chunk_index, output_dir, field_names)

    Returns
    -------
    str
        Path to the written part file.
    """
    input_ms, startrow, nrow, chunk_index, output_dir, field_names = args

    _tb = table()
    _tb.open(input_ms)

    # Read columns with row range
    uvw = _tb.getcol("UVW", startrow=startrow, nrow=nrow)  # shape (3, nrow)
    ant1 = _tb.getcol("ANTENNA1", startrow=startrow, nrow=nrow)
    ant2 = _tb.getcol("ANTENNA2", startrow=startrow, nrow=nrow)
    time_col = _tb.getcol("TIME", startrow=startrow, nrow=nrow)
    flag = _tb.getcol("FLAG", startrow=startrow, nrow=nrow)  # (npol, nchan, nrow)
    data = _tb.getcol("DATA", startrow=startrow, nrow=nrow)  # (npol, nchan, nrow)
    ddid = _tb.getcol("DATA_DESC_ID", startrow=startrow, nrow=nrow)
    field_id = _tb.getcol("FIELD_ID", startrow=startrow, nrow=nrow)

    _tb.close()

    # Map field IDs to human-readable names
    field_col = [field_names[fid] for fid in field_id]

    # Compute amplitude and phase with flag masking
    # data shape: (npol, nchan, nrow), flag shape: (npol, nchan, nrow)
    amplitude = np.abs(data)  # (npol, nchan, nrow)
    phase = np.angle(data)    # (npol, nchan, nrow)

    # Mask flagged values with NaN
    amplitude = np.where(~flag, amplitude, np.nan)
    phase = np.where(~flag, phase, np.nan)

    # Mean over pol and chan axes -> (nrow,)
    with np.errstate(invalid="ignore"):
        amp = np.nanmean(amplitude, axis=(0, 1))
        pha = np.nanmean(phase, axis=(0, 1))

    # UV distance
    uvdist = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)

    result = {
        "ANTENNA1": ant1.astype(np.int32),
        "ANTENNA2": ant2.astype(np.int32),
        "U": uvw[0],
        "V": uvw[1],
        "W": uvw[2],
        "AMP": amp,
        "PHASE": pha,
        "UVDIST": uvdist,
        "TIME": time_col,
        "SPW_ID": ddid.astype(np.int32),
        "FIELD": field_col,
    }

    part_table = pa.Table.from_pydict(result)
    part_path = os.path.join(output_dir, f"part.{chunk_index}.parquet")
    pq.write_table(part_table, part_path, compression="zstd")

    return part_path


def ms_to_parquet(
    input_ms: str,
    output_pq: str = None,
    nworkers: int = None,
    npartitions: int = None,
    overwrite: bool = False,
    max_mem: float = None,
) -> str:
    """
    Parallel read of MS to partitioned parquet directory.

    Parameters
    ----------
    input_ms : str
        Path to the measurement set.
    output_pq : str, optional
        Output parquet directory path. Defaults to input_ms + ".pq"
    nworkers : int, optional
        Number of parallel workers. Defaults to cpu_count().
    npartitions : int, optional
        Number of parquet partitions. Auto-sized from ``max_mem`` when not
        given, so that all workers fit within the memory budget.
    overwrite : bool
        If True, overwrite existing parquet directory.
    max_mem : float, optional
        Maximum total RAM budget in GB for parallel workers. Used to
        auto-size ``npartitions`` so each chunk's peak working set fits in
        ``max_mem / nworkers``. Defaults to total system RAM.

    Returns
    -------
    str
        Path to the written parquet directory.
    """
    if output_pq is None:
        output_pq = input_ms + ".pq"

    if os.path.exists(output_pq):
        if not overwrite:
            print(f"{output_pq} exists, not overwriting.")
            return output_pq
        if os.path.isdir(output_pq):
            shutil.rmtree(output_pq)
        else:
            os.remove(output_pq)

    if nworkers is None:
        nworkers = cpu_count()

    meta = get_ms_metadata(input_ms)

    # Build field ID → name mapping
    _msmd = msmetadata()
    _msmd.open(input_ms)
    field_names = _msmd.fieldnames()
    _msmd.close()

    if npartitions is None:
        max_nchan = max(meta["nchan"].values())
        # Peak memory per row: DATA + FLAG + amp + phase + masked intermediates
        bytes_per_row = meta["npol"] * max_nchan * 50
        if max_mem is None:
            max_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        max_mem_bytes = max_mem * 1024**3
        mem_per_worker = max_mem_bytes / nworkers
        rows_per_chunk = max(1, int(mem_per_worker / bytes_per_row))
        npartitions = max(nworkers, math.ceil(meta["nrow"] / rows_per_chunk))
        print(
            f"Memory budget: {max_mem:.1f} GB total, "
            f"~{mem_per_worker / 1024**3:.1f} GB/worker → "
            f"{rows_per_chunk} rows/chunk"
        )

    # Create output directory
    os.makedirs(output_pq, exist_ok=True)

    # Compute chunks (decoupled from nworkers)
    chunks = compute_time_chunks(input_ms, npartitions)
    print(f"Reading {meta['nrow']} rows in {len(chunks)} chunks with {nworkers} workers")

    # Prepare worker args — each chunk gets its index and output dir
    worker_args = [
        (input_ms, startrow, nrow, i, output_pq, field_names)
        for i, (startrow, nrow) in enumerate(chunks)
    ]

    # Parallel read — workers write part files directly
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(nworkers) as pool:
        part_paths = pool.map(_read_chunk, worker_args)

    total_rows = sum(pq.read_metadata(p).num_rows for p in part_paths)
    print(f"Wrote {total_rows} rows across {len(part_paths)} part files to {output_pq}")
    return output_pq


def get_spw_freqs(input_ms: str):
    """
    Given an input MS, extract the spectral window frequencies.

    Parameters
    ----------
    input_ms : str
        The path to the measurement set.

    Returns
    -------
    spwfreqs
        An array of spectral window frequencies.
    """
    _msmd = msmetadata()
    _msmd.open(input_ms)
    nspw = _msmd.nspw()

    spwfreqs = {}
    reffreqs = {}

    for spw in range(nspw):
        freqs = _msmd.chanfreqs(spw)
        spwfreqs[spw] = freqs
        reffreqs[spw] = _msmd.reffreq(spw)["m0"]["value"]

    _msmd.close()

    return spwfreqs, reffreqs


def scale_uvw_fequency(uvw: np.ndarray, freqs: np.ndarray, reffreqs: np.ndarray):
    """
    Scale the UVW coordinates by the frequency.

    Parameters
    ----------
    uvw : np.ndarray
        The UVW coordinates.
    freq : np.ndarray
        The frequency values per channel and spectral window
    reffreqs : np.ndarray
        The reference frequencies for each spectral window.

    Returns
    -------
    uvw_scaled : np.ndarray
        The scaled UVW coordinates.
    """

    spw_list = freqs.keys()

    for spw in spw_list:
        frac_freqs = 1 + (freqs[spw] - reffreqs[spw]) / reffreqs[spw]
        nfreq = len(frac_freqs)

        scaled_uvw = (
            uvw[:, :, np.newaxis] * frac_freqs[np.newaxis, np.newaxis, :]
        ).reshape(uvw.shape[0], nfreq, -1)
        print(np.count_nonzero(scaled_uvw))

    print(uvw.shape)
    print(scaled_uvw.shape)
    print(uvw[0, 100])
    print(scaled_uvw[0, 16 * 100])
    print(freqs[0], reffreqs[0])
