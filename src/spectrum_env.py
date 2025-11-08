import numpy as np

from .config import (
    DATA_PATH,
    N_FFT,
    ALPHA_RATE,
    BETA_INT,
    STEPS_PER_EPISODE,
)
from .data_loader import load_iq_h5, segment_iq
from .spectrum_utils import compute_psd_db, split_subbands, band_power
from .sic import two_stage_sic


class SpectrumEnv:
    """
    Hybrid environment that exposes spectrum observations for the DDQN agent.
    Each step processes one IQ segment, generates spectral features, executes SIC,
    and returns reward signals that balance active throughput and passive protection.
    """

    def __init__(self, seg_len: int = 4096, hop: int = 4096):
        iq = load_iq_h5(DATA_PATH)
        self.segments = segment_iq(iq, seg_len, hop)
        self.num_segments = self.segments.shape[0]
        self.seg_len = seg_len
        self.ptr = 0

        self.bands = split_subbands(np.zeros(N_FFT))
        self.state_dim = len(self.bands)

    def reset(self) -> np.ndarray:
        self.ptr = np.random.randint(0, max(1, self.num_segments - STEPS_PER_EPISODE))
        return self._build_state()

    def _build_state(self) -> np.ndarray:
        seg = self.segments[self.ptr]
        psd = compute_psd_db(seg)
        band_powers = np.array([band_power(psd, b) for b in self.bands], dtype=np.float32)
        return (band_powers - band_powers.mean()) / (band_powers.std() + 1e-8)

    def step(self, action: int):
        seg = self.segments[self.ptr]
        band = self.bands[action]

        x_hat, cleaned = two_stage_sic(seg, band)

        sig_power = np.mean(np.abs(x_hat) ** 2)
        residual_power = np.mean(np.abs(cleaned) ** 2) + 1e-12
        snr_active = 10 * np.log10(sig_power / residual_power + 1e-12)

        psd_after = compute_psd_db(cleaned)
        interference_band_power = band_power(psd_after, band)

        reward = ALPHA_RATE * snr_active - BETA_INT * interference_band_power

        self.ptr += 1
        done = self.ptr >= self.num_segments - 1

        next_state = self._build_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "snr_active": float(snr_active),
            "interference_band_power": float(interference_band_power),
        }
        return next_state, float(reward), done, info
