"""
IO module
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from casatools import ms as measurementset
from casatools import msmetadata, table

ms = measurementset()
msmd = msmetadata()
tb = table()


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

    msmd.open(input_ms)
    nspw = msmd.nspw()

    spwfreqs = {}
    reffreqs = {}

    for spw in range(nspw):
        freqs = msmd.chanfreqs(spw)
        spwfreqs[spw] = freqs
        reffreqs[spw] = msmd.reffreq(spw)["m0"]["value"]

    msmd.close()

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


def ms_to_df(input_ms: str, persist: bool = False, overwrite: bool = False):
    """
    Given an input MS, extract the UV values, amplitudes and meta-data into
    a polars lazyframe (dataframe).

    Optionally if persist is True, it writes the dataframe to disk in parquet format.

    Parameters
    ----------
    input_ms : str
        The path to the measurement set.
    persist : bool
        If True, persist the dataframe to disk in parquet format.
    overwrite : bool
        If True, overwrites the dataframe on disk if it exists.

    Returns
    -------
    pl.LazyFrame
        A polars lazyframe containing the UV values, amplitudes and meta-data.
    """

    pqname = input_ms + ".parquet"
    if overwrite is False and os.path.exists(pqname):
        print(f"{pqname} exists, not overwriting.")
        uvdf = pd.read_parquet(pqname)
    else:
        ms.open(input_ms)
        ms.selectinit(reset=True)

        spwinfo = ms.getspectralwindowinfo()
        spwfreqs, reffreqs = get_spw_freqs(input_ms)

        cols = ["UVW", "DATA", "ANTENNA1", "ANTENNA2", "FLAG"]

        # ifraxis separates out data into interferometer axis and time
        # This is slow and not necessary here
        # TODO : Need to work on an iterator here
        msdata = ms.getdata(items=cols, ifraxis=False)

        # uvw_freq = scale_uvw_fequency(msdata['uvw'], spwfreqs, reffreqs)

        for key in msdata.keys():
            print(key, msdata[key].shape)

        npol = msdata["data"].shape[0]
        nchan = msdata["data"].shape[1]

        uvdf = pd.DataFrame(
            {
                "U": np.repeat(msdata["uvw"][0], npol * nchan),
                "V": np.repeat(msdata["uvw"][1], npol * nchan),
                "W": np.repeat(msdata["uvw"][2], npol * nchan),
                "AMP": np.abs(msdata["data"]).ravel(),
                "PHASE": np.angle(msdata["data"]).ravel(),
                "ANTENNA1": np.repeat(msdata["antenna1"], npol * nchan),
                "ANTENNA2": np.repeat(msdata["antenna2"], npol * nchan),
                "FLAG": msdata["flag"].ravel(),
            }
        )

        if persist:
            pq.write_to_dataset(
                pa.Table.from_pandas(uvdf, preserve_index=False),
                root_path=pqname,
                # partition_cols=['ANTENNA1', 'ANTENNA2'],
                compression="zstd",
                filesystem=None,
            )

        ms.close()

    if not persist:
        pqname = None

    return uvdf, pqname
