import numpy as np


def reconstruct_active_sic1(iq: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    """
    Stage 1 SIC: coarse estimate of the active 5G component within the selected band.
    This placeholder performs a band-limited reconstruction via FFT masking.
    """
    start, end = band
    n = len(iq)
    spectrum = np.fft.fft(iq, n=n)
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[start:end] = True
    active_spec = np.where(mask, spectrum, 0.0)
    return np.fft.ifft(active_spec)


def reconstruct_active_sic2(residual: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    """
    Stage 2 SIC: refine the active signal estimate using the residual.
    This placeholder re-applies band filtering; replace with advanced demod/channel est.
    """
    start, end = band
    n = len(residual)
    spectrum = np.fft.fft(residual, n=n)
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[start:end] = True
    refined_spec = np.where(mask, spectrum, 0.0)
    return np.fft.ifft(refined_spec)


def two_stage_sic(iq: np.ndarray, band: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute the two-stage SIC procedure.

    Returns:
        x_hat_total: combined reconstructed active signal (SIC1 + SIC2).
        cleaned: residual signal after subtracting the reconstructed active component.
    """
    x1 = reconstruct_active_sic1(iq, band)
    residual1 = iq - x1
    x2 = reconstruct_active_sic2(residual1, band)
    x_hat_total = x1 + x2
    cleaned = iq - x_hat_total
    return x_hat_total, cleaned
