import numpy as np

from .config import N_FFT, N_SUBBANDS


def compute_psd_db(iq: np.ndarray) -> np.ndarray:
    """
    Compute the power spectral density (PSD) in dB for an IQ segment.
    Uses a Hann window and FFT length N_FFT.
    """
    window = np.hanning(len(iq))
    spectrum = np.fft.fftshift(np.fft.fft(iq * window, n=N_FFT))
    psd = np.abs(spectrum) ** 2
    psd /= psd.max() + 1e-12
    return 10 * np.log10(psd + 1e-12)


def split_subbands(psd: np.ndarray) -> list[tuple[int, int]]:
    """
    Divide the spectrum into N_SUBBANDS equal-width bands.
    Returns a list of (start_idx, end_idx) tuples.
    """
    length = len(psd)
    band_len = length // N_SUBBANDS
    bands = []
    for b in range(N_SUBBANDS):
        start = b * band_len
        end = (b + 1) * band_len if b < N_SUBBANDS - 1 else length
        bands.append((start, end))
    return bands


def band_power(psd: np.ndarray, band: tuple[int, int]) -> float:
    """
    Compute the average linear power over a PSD band specified by indices.
    """
    start, end = band
    return float(np.mean(10 ** (psd[start:end] / 10.0)))
