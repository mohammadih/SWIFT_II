import numpy as np
import tensorflow as tf

from .config import N_FFT, DATA_PATH
from .data_loader import load_iq_h5, segment_iq
from .spectrum_utils import compute_psd_db, split_subbands, band_power
from .sic import two_stage_sic


def main():
    iq = load_iq_h5(DATA_PATH)
    segments = segment_iq(iq, seg_len=4096, hop=4096)

    model = tf.keras.models.load_model("ddqn_spectrum_agent.h5")
    bands = split_subbands(np.zeros(N_FFT))

    cleaned_segments = []

    for seg in segments:
        psd = compute_psd_db(seg)
        band_powers = np.array([band_power(psd, b) for b in bands], dtype=np.float32)
        band_powers = (band_powers - band_powers.mean()) / (band_powers.std() + 1e-8)

        q_values = model(np.expand_dims(band_powers, axis=0), training=False)
        action = int(tf.argmax(q_values[0]).numpy())
        band = bands[action]

        _, cleaned = two_stage_sic(seg, band)
        cleaned_segments.append(cleaned)

    cleaned_signal = np.concatenate(cleaned_segments)
    np.save("cleaned_signal.npy", cleaned_signal.astype(np.complex64))
    print("Saved cleaned_signal.npy")


if __name__ == "__main__":
    main()
