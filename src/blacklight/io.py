"""
IO module
"""

import os
from multiprocessing import Pool, cpu_count

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


def compute_time_chunks(input_ms: str, nworkers: int = None) -> list[tuple[int, int]]:
    """
    Compute row ranges partitioned by time boundaries.

    Splits the MS into approximately nworkers chunks, respecting integration
    boundaries (won't split mid-integration).

    Parameters
    ----------
    input_ms : str
        Path to the measurement set.
    nworkers : int, optional
        Number of chunks to create. Defaults to cpu_count().

    Returns
    -------
    list[tuple[int, int]]
        List of (startrow, nrow) tuples.
    """
    if nworkers is None:
        nworkers = cpu_count()

    _tb = table()
    _tb.open(input_ms)
    time_col = _tb.getcol("TIME")
    nrow = _tb.nrows()
    _tb.close()

    # Find unique time values and their first occurrence indices
    unique_times, first_indices = np.unique(time_col, return_index=True)
    n_times = len(unique_times)

    if n_times <= nworkers:
        # More workers than time steps - one chunk per time step
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

    # Distribute time steps across workers
    times_per_worker = n_times // nworkers
    chunks = []
    sorted_indices = np.argsort(first_indices)

    for w in range(nworkers):
        start_time_idx = w * times_per_worker
        if w == nworkers - 1:
            # Last worker gets remaining
            end_time_idx = n_times
        else:
            end_time_idx = (w + 1) * times_per_worker

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


def _read_chunk(args: tuple) -> pa.Table:
    """
    Worker function: read a row range from MS and return Arrow table.

    Parameters
    ----------
    args : tuple
        (input_ms, startrow, nrow, read_corrected)

    Returns
    -------
    pa.Table
        Arrow table with chunk data.
    """
    input_ms, startrow, nrow, read_corrected = args

    _tb = table()
    _tb.open(input_ms)

    # Read columns with row range
    uvw = _tb.getcol("UVW", startrow=startrow, nrow=nrow)  # shape (3, nrow)
    ant1 = _tb.getcol("ANTENNA1", startrow=startrow, nrow=nrow)
    ant2 = _tb.getcol("ANTENNA2", startrow=startrow, nrow=nrow)
    time_col = _tb.getcol("TIME", startrow=startrow, nrow=nrow)
    flag = _tb.getcol("FLAG", startrow=startrow, nrow=nrow)  # (npol, nchan, nrow)
    data = _tb.getcol("DATA", startrow=startrow, nrow=nrow)  # (npol, nchan, nrow)

    # DATA_DESC_ID maps to SPW
    ddid = _tb.getcol("DATA_DESC_ID", startrow=startrow, nrow=nrow)

    corrected_real = None
    corrected_imag = None
    if read_corrected:
        corr = _tb.getcol("CORRECTED_DATA", startrow=startrow, nrow=nrow)
        corrected_real = corr.real
        corrected_imag = corr.imag

    _tb.close()

    # Convert arrays to list-of-lists format for parquet
    # DATA shape: (npol, nchan, nrow) -> per-row: list of npol lists, each with nchan values
    npol, nchan, chunk_nrow = data.shape

    # Reshape: (npol, nchan, nrow) -> (nrow, npol, nchan) -> list of lists
    data_real_arr = data.real.transpose(2, 0, 1)  # (nrow, npol, nchan)
    data_imag_arr = data.imag.transpose(2, 0, 1)
    flag_arr = flag.transpose(2, 0, 1)

    # Convert to nested lists
    data_real_nested = [row.tolist() for row in data_real_arr]
    data_imag_nested = [row.tolist() for row in data_imag_arr]
    flag_nested = [row.tolist() for row in flag_arr]

    result = {
        "ANTENNA1": ant1.astype(np.int32),
        "ANTENNA2": ant2.astype(np.int32),
        "U": uvw[0],
        "V": uvw[1],
        "W": uvw[2],
        "DATA_REAL": data_real_nested,
        "DATA_IMAG": data_imag_nested,
        "FLAG": flag_nested,
        "TIME": time_col,
        "SPW_ID": ddid.astype(np.int32),
    }

    if read_corrected:
        corr_real_arr = corrected_real.transpose(2, 0, 1)
        corr_imag_arr = corrected_imag.transpose(2, 0, 1)
        result["CORRECTED_REAL"] = [row.tolist() for row in corr_real_arr]
        result["CORRECTED_IMAG"] = [row.tolist() for row in corr_imag_arr]

    return pa.Table.from_pydict(result)


def ms_to_parquet(
    input_ms: str,
    output_pq: str = None,
    nworkers: int = None,
    overwrite: bool = False,
    include_corrected: bool = True,
) -> str:
    """
    Parallel read of MS to parquet file.

    Parameters
    ----------
    input_ms : str
        Path to the measurement set.
    output_pq : str, optional
        Output parquet path. Defaults to input_ms + ".parquet"
    nworkers : int, optional
        Number of parallel workers. Defaults to cpu_count().
    overwrite : bool
        If True, overwrite existing parquet file.
    include_corrected : bool
        If True and CORRECTED_DATA exists, include it.

    Returns
    -------
    str
        Path to the written parquet file.
    """
    if output_pq is None:
        output_pq = input_ms + ".parquet"

    if not overwrite and os.path.exists(output_pq):
        print(f"{output_pq} exists, not overwriting.")
        return output_pq

    if nworkers is None:
        nworkers = cpu_count()

    # Get metadata to check for CORRECTED_DATA
    meta = get_ms_metadata(input_ms)
    read_corrected = include_corrected and meta["has_corrected"]

    # Compute chunks
    chunks = compute_time_chunks(input_ms, nworkers)
    print(f"Reading {meta['nrow']} rows in {len(chunks)} chunks with {nworkers} workers")

    # Prepare worker args
    worker_args = [
        (input_ms, startrow, nrow, read_corrected) for startrow, nrow in chunks
    ]

    # Parallel read
    with Pool(nworkers) as pool:
        tables = pool.map(_read_chunk, worker_args)

    # Concatenate and write
    combined = pa.concat_tables(tables)
    pq.write_table(combined, output_pq, compression="zstd")

    print(f"Wrote {combined.num_rows} rows to {output_pq}")
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


