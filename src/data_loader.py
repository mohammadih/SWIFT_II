import numpy as np
import h5py
from typing import Tuple


def load_iq_h5(path: str) -> np.ndarray:
    """
    Load complex IQ samples from an HDF5 file.
    The dataset is expected to store IQ either as complex numbers or as two real columns.
    """
    with h5py.File(path, "r") as f:
        dataset_name = list(f.keys())[0]
        dset = f[dataset_name]
        iq = dset[()]  # Could be (N, 2) real or complex-valued

    if iq.ndim == 2 and iq.shape[1] == 2:
        iq = iq[:, 0] + 1j * iq[:, 1]

    return iq.astype(np.complex64)


def segment_iq(iq: np.ndarray, seg_len: int, hop: int) -> np.ndarray:
    """
    Create overlapping IQ segments of a given length and hop size.
    Returns an array of shape (num_segments, seg_len).
    """
    n = len(iq)
    indices = range(0, n - seg_len + 1, hop)
    segments = np.stack([iq[i:i + seg_len] for i in indices], axis=0)
    return segments
