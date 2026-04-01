import librosa
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Loading / preprocessing
# -----------------------------
def load_motor_segment(file_path, start, end):
    """
    Load an audio file and return the chosen segment.
    """
    samples, sample_rate = librosa.load(file_path, sr=None, mono=True)

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = samples[start_sample:end_sample]

    time = np.linspace(start, end, len(segment), endpoint=False)
    return segment, sample_rate, time


def compute_fft(segment, sample_rate):
    """
    Compute FFT magnitude spectrum for a signal segment.
    Uses a Hann window to reduce spectral leakage.
    """
    if len(segment) == 0:
        raise ValueError("Segment is empty. Check start/end times.")

    window = np.hanning(len(segment))
    windowed_segment = segment * window

    fft_vals = np.fft.rfft(windowed_segment)
    fft_freqs = np.fft.rfftfreq(len(windowed_segment), d=1 / sample_rate)
    fft_mag = np.abs(fft_vals)

    return fft_freqs, fft_mag


# -----------------------------
# Plotting
# -----------------------------
def plot_motor_segment(time, segment, title, ax):
    ax.plot(time, segment)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)


def plot_motor_fft(fft_freqs, fft_mag, title, ax, max_freq=5000):
    ax.plot(fft_freqs, fft_mag)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title + " FFT Spectrum")
    ax.set_xlim(0, max_freq)
    ax.grid(True)


# -----------------------------
# Peak / feature utilities
# -----------------------------
def get_amplitude_near_frequency(fft_freqs, fft_mag, target_freq, tolerance_hz=20):
    """
    Return the largest FFT magnitude within +/- tolerance_hz of target_freq.
    """
    mask = (fft_freqs >= target_freq - tolerance_hz) & (fft_freqs <= target_freq + tolerance_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(fft_mag[mask]))


def find_fundamental_frequency(fft_freqs, fft_mag, min_freq=100, max_freq=3000):
    """
    Beta version:
    Find the strongest peak in a specified range and treat it as the fundamental.
    """
    mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
    if not np.any(mask):
        raise ValueError("No FFT points found in the requested frequency range.")

    masked_freqs = fft_freqs[mask]
    masked_mag = fft_mag[mask]

    peak_index = np.argmax(masked_mag)
    fundamental_freq = float(masked_freqs[peak_index])
    fundamental_amp = float(masked_mag[peak_index])

    return fundamental_freq, fundamental_amp


def get_harmonic_amplitudes(fft_freqs, fft_mag, fundamental_freq, num_harmonics=5, tolerance_hz=20):
    """
    Return amplitudes at 1X, 2X, 3X, ... harmonics.
    """
    harmonic_amps = []
    harmonic_freqs = []

    for k in range(1, num_harmonics + 1):
        harmonic_freq = k * fundamental_freq
        amp = get_amplitude_near_frequency(fft_freqs, fft_mag, harmonic_freq, tolerance_hz)
        harmonic_freqs.append(harmonic_freq)
        harmonic_amps.append(amp)

    return harmonic_freqs, harmonic_amps


def count_significant_peaks(
    fft_freqs,
    fft_mag,
    min_freq=100,
    max_freq=5000,
    threshold_ratio=0.10,
    min_spacing_hz=80
):
    """
    Count local maxima above a threshold.
    threshold_ratio is relative to the maximum magnitude in the selected band.
    """
    mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
    freqs = fft_freqs[mask]
    mags = fft_mag[mask]

    if len(freqs) < 3:
        return 0, [], []

    threshold = threshold_ratio * np.max(mags)

    peak_freqs = []
    peak_amps = []

    last_kept_freq = -1e9

    for i in range(1, len(mags) - 1):
        is_local_peak = mags[i] > mags[i - 1] and mags[i] > mags[i + 1]
        is_big_enough = mags[i] >= threshold
        far_enough = (freqs[i] - last_kept_freq) >= min_spacing_hz

        if is_local_peak and is_big_enough and far_enough:
            peak_freqs.append(float(freqs[i]))
            peak_amps.append(float(mags[i]))
            last_kept_freq = freqs[i]

    return len(peak_freqs), peak_freqs, peak_amps


def compute_band_energy(fft_freqs, fft_mag, low_freq, high_freq):
    """
    Simple spectral energy measure in a frequency band.
    """
    mask = (fft_freqs >= low_freq) & (fft_freqs <= high_freq)
    if not np.any(mask):
        return 0.0
    return float(np.sum(fft_mag[mask] ** 2))


def extract_motor_features(fft_freqs, fft_mag):
    """
    Build a feature dictionary for one motor.
    """
    fundamental_freq, fundamental_amp = find_fundamental_frequency(
        fft_freqs, fft_mag, min_freq=100, max_freq=3000
    )

    harmonic_freqs, harmonic_amps = get_harmonic_amplitudes(
        fft_freqs, fft_mag, fundamental_freq, num_harmonics=5, tolerance_hz=25
    )

    num_peaks, peak_freqs, peak_amps = count_significant_peaks(
        fft_freqs,
        fft_mag,
        min_freq=100,
        max_freq=5000,
        threshold_ratio=0.10,
        min_spacing_hz=80
    )

    low_band_energy = compute_band_energy(fft_freqs, fft_mag, 0, 1000)
    mid_band_energy = compute_band_energy(fft_freqs, fft_mag, 1000, 3000)
    high_band_energy = compute_band_energy(fft_freqs, fft_mag, 3000, 5000)

    h1 = harmonic_amps[0] if len(harmonic_amps) > 0 else 0.0
    h2 = harmonic_amps[1] if len(harmonic_amps) > 1 else 0.0
    h3 = harmonic_amps[2] if len(harmonic_amps) > 2 else 0.0

    h2_to_h1 = h2 / h1 if h1 > 0 else 0.0
    h3_to_h1 = h3 / h1 if h1 > 0 else 0.0

    return {
        "fundamental_freq": fundamental_freq,
        "fundamental_amp": fundamental_amp,
        "harmonic_freqs": harmonic_freqs,
        "harmonic_amps": harmonic_amps,
        "num_significant_peaks": num_peaks,
        "peak_freqs": peak_freqs,
        "peak_amps": peak_amps,
        "low_band_energy": low_band_energy,
        "mid_band_energy": mid_band_energy,
        "high_band_energy": high_band_energy,
        "h2_to_h1": h2_to_h1,
        "h3_to_h1": h3_to_h1,
    }


# -----------------------------
# Fault detection
# -----------------------------
def detect_imbalance(features, baseline_features):
    """
    Imbalance logic:
    - 1X amplitude noticeably larger than baseline
    - harmonic complexity still relatively low
    - not too many extra peaks
    """
    reasons = []

    fundamental_ratio = (
        features["fundamental_amp"] / baseline_features["fundamental_amp"]
        if baseline_features["fundamental_amp"] > 0
        else 0.0
    )

    low_harmonic_complexity = (features["h2_to_h1"] < 0.20) and (features["h3_to_h1"] < 0.15)
    modest_peak_count = features["num_significant_peaks"] <= baseline_features["num_significant_peaks"] + 1
    strong_1x_increase = fundamental_ratio >= 1.03

    if strong_1x_increase:
        reasons.append(f"fundamental amplitude ratio vs baseline = {fundamental_ratio:.2f} (high)")
    else:
        reasons.append(f"fundamental amplitude ratio vs baseline = {fundamental_ratio:.2f} (not high enough)")

    reasons.append(
        f"H2/H1 = {features['h2_to_h1']:.3f}, H3/H1 = {features['h3_to_h1']:.3f}"
    )
    reasons.append(
        f"significant peak count = {features['num_significant_peaks']} "
        f"(baseline {baseline_features['num_significant_peaks']})"
    )

    detected = strong_1x_increase and low_harmonic_complexity and modest_peak_count

    if detected:
        summary = "Strong 1X increase with limited harmonic complexity."
    else:
        summary = "Imbalance pattern not strong enough."

    return {
        "detected": detected,
        "summary": summary,
        "reasons": reasons,
    }


def detect_misalignment_or_looseness(features, baseline_features):
    """
    Misalignment / looseness logic:
    - more significant peaks than baseline
    - higher 2X / 3X relative harmonic content
    - higher spread energy in mid/high bands
    """
    reasons = []

    peak_count_increase = (
        features["num_significant_peaks"] >= baseline_features["num_significant_peaks"] + 2
    )

    harmonic_growth = (
        features["h2_to_h1"] >= baseline_features["h2_to_h1"] * 1.5
        or features["h3_to_h1"] >= baseline_features["h3_to_h1"] * 1.5
    )

    mid_energy_ratio = (
        features["mid_band_energy"] / baseline_features["mid_band_energy"]
        if baseline_features["mid_band_energy"] > 0
        else 0.0
    )

    high_energy_ratio = (
        features["high_band_energy"] / baseline_features["high_band_energy"]
        if baseline_features["high_band_energy"] > 0
        else 0.0
    )

    elevated_spread_energy = (mid_energy_ratio >= 1.50) or (high_energy_ratio >= 3.00)

    reasons.append(
        f"significant peak count = {features['num_significant_peaks']} "
        f"(baseline {baseline_features['num_significant_peaks']})"
    )
    reasons.append(
        f"H2/H1 = {features['h2_to_h1']:.3f} (baseline {baseline_features['h2_to_h1']:.3f}), "
        f"H3/H1 = {features['h3_to_h1']:.3f} (baseline {baseline_features['h3_to_h1']:.3f})"
    )
    reasons.append(
        f"mid-band energy ratio = {mid_energy_ratio:.2f}, "
        f"high-band energy ratio = {high_energy_ratio:.2f}"
    )

    score = int(peak_count_increase) + int(harmonic_growth) + int(elevated_spread_energy)
    detected = score >= 2

    if detected:
        summary = "More harmonics / extra peaks / spread spectral energy than baseline."
    else:
        summary = "Misalignment/looseness pattern not strong enough."

    return {
        "detected": detected,
        "summary": summary,
        "reasons": reasons,
    }


def diagnose_motor(motor_name, features, baseline_features):
    imbalance_result = detect_imbalance(features, baseline_features)
    misalignment_result = detect_misalignment_or_looseness(features, baseline_features)

    return {
        "motor_name": motor_name,
        "features": features,
        "imbalance": imbalance_result,
        "misalignment_looseness": misalignment_result,
    }


def print_feature_summary(motor_name, features):
    print(f"\n=== {motor_name} Features ===")
    print(f"Fundamental frequency: {features['fundamental_freq']:.2f} Hz")
    print(f"Fundamental amplitude: {features['fundamental_amp']:.2f}")
    print(f"Harmonic amplitudes: {[round(x, 2) for x in features['harmonic_amps']]}")
    print(f"Significant peak count: {features['num_significant_peaks']}")
    print(f"Low-band energy (0-1000 Hz): {features['low_band_energy']:.2f}")
    print(f"Mid-band energy (1000-3000 Hz): {features['mid_band_energy']:.2f}")
    print(f"High-band energy (3000-5000 Hz): {features['high_band_energy']:.2f}")


def print_diagnosis_report(result):
    print(f"\n==============================")
    print(f"{result['motor_name']} Diagnosis")
    print(f"==============================")

    imbalance = result["imbalance"]
    misalignment = result["misalignment_looseness"]

    print(f"\nImbalance: {'DETECTED' if imbalance['detected'] else 'NOT DETECTED'}")
    print(f"Summary: {imbalance['summary']}")
    for reason in imbalance["reasons"]:
        print(f"  - {reason}")

    print(f"\nMisalignment / Looseness: {'DETECTED' if misalignment['detected'] else 'NOT DETECTED'}")
    print(f"Summary: {misalignment['summary']}")
    for reason in misalignment["reasons"]:
        print(f"  - {reason}")


# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    # File paths
    motor_files = {
        "Motor 1": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_1_O.mp3",
        "Motor 3": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_3_O.mp3",
        "Motor 4": r"C:\Users\Dell\Desktop\Code\University\ESC190\Praxis\Motor_4_O.mp3",
    }

    start_time = 2
    end_time = 8

    motor_data = {}

    # Load, FFT, and extract features
    for motor_name, file_path in motor_files.items():
        segment, sample_rate, time = load_motor_segment(file_path, start_time, end_time)
        fft_freqs, fft_mag = compute_fft(segment, sample_rate)
        features = extract_motor_features(fft_freqs, fft_mag)

        motor_data[motor_name] = {
            "segment": segment,
            "sample_rate": sample_rate,
            "time": time,
            "fft_freqs": fft_freqs,
            "fft_mag": fft_mag,
            "features": features,
        }

    # Motor 1 is the healthy baseline
    baseline_features = motor_data["Motor 1"]["features"]

    # Print features
    print_feature_summary("Motor 1 (Baseline)", motor_data["Motor 1"]["features"])
    print_feature_summary("Motor 3", motor_data["Motor 3"]["features"])
    print_feature_summary("Motor 4", motor_data["Motor 4"]["features"])

    # Diagnose motors against baseline
    motor3_result = diagnose_motor("Motor 3", motor_data["Motor 3"]["features"], baseline_features)
    motor4_result = diagnose_motor("Motor 4", motor_data["Motor 4"]["features"], baseline_features)

    # Print reports
    print_diagnosis_report(motor3_result)
    print_diagnosis_report(motor4_result)

    # Plot time-domain segments
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10))
    plot_motor_segment(
        motor_data["Motor 1"]["time"],
        motor_data["Motor 1"]["segment"],
        "Motor 1",
        axs1[0]
    )
    plot_motor_segment(
        motor_data["Motor 3"]["time"],
        motor_data["Motor 3"]["segment"],
        "Motor 3",
        axs1[1]
    )
    plot_motor_segment(
        motor_data["Motor 4"]["time"],
        motor_data["Motor 4"]["segment"],
        "Motor 4",
        axs1[2]
    )
    plt.tight_layout()
    plt.show()

    # Plot FFT spectra
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 10))
    plot_motor_fft(
        motor_data["Motor 1"]["fft_freqs"],
        motor_data["Motor 1"]["fft_mag"],
        "Motor 1",
        axs2[0]
    )
    plot_motor_fft(
        motor_data["Motor 3"]["fft_freqs"],
        motor_data["Motor 3"]["fft_mag"],
        "Motor 3",
        axs2[1]
    )
    plot_motor_fft(
        motor_data["Motor 4"]["fft_freqs"],
        motor_data["Motor 4"]["fft_mag"],
        "Motor 4",
        axs2[2]
    )
    plt.tight_layout()
    plt.show()