import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# ─────────────────────────────────────────────
#  CORE SIGNAL LOADING
# ─────────────────────────────────────────────

def load_segment(file_path, start, end):
    """Load an audio file and return the trimmed segment + sample rate."""
    samples, sr = librosa.load(file_path, sr=None)
    return samples[int(start * sr):int(end * sr)], sr


# ─────────────────────────────────────────────
#  WELCH PSD  (averaged, windowed, in dB)
# ─────────────────────────────────────────────

def compute_welch_psd(segment, sr, nperseg=4096):
    """
    Compute a Welch-averaged Power Spectral Density.
    Splits signal into overlapping Hann-windowed chunks, computes
    FFT of each, averages the squared magnitudes.
    Returns:
        freqs : frequency axis (Hz)
        psd_db: power in dB
    """
    freqs, psd = welch(segment, fs=sr, window='hann',
                       nperseg=nperseg, noverlap=nperseg // 2,
                       scaling='density')
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, psd_db


# ─────────────────────────────────────────────
#  BASELINE MANAGEMENT
# ─────────────────────────────────────────────

def save_baseline(freqs, psd_db, out_path):
    np.savez(out_path, freqs=freqs, psd_db=psd_db)
    print(f"Baseline saved → {out_path}")


def load_baseline(npz_path):
    data = np.load(npz_path)
    return data['freqs'], data['psd_db']


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def plot_time(segment, sr, start, title):
    time = np.linspace(start, start + len(segment) / sr, len(segment))
    plt.figure(figsize=(12, 4))
    plt.plot(time, segment, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{title} — Time Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_psd(freqs, psd_db, title, fmax=10000):
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, psd_db, linewidth=0.8)
    plt.xlim(0, fmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"{title} — Welch PSD (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    MOTORS = {
        "Motor 1": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_1_O.mp3",
        "Motor 3": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_3_O.mp3",
        "Motor 4": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_4_O.mp3",
    }

    for name, path in MOTORS.items():
        segment, sr = load_segment(path, start=2, end=8)
        freqs, psd_db = compute_welch_psd(segment, sr)

        plot_time(segment, sr, start=2, title=name)
        plot_psd(freqs, psd_db, title=name)