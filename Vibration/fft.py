import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("gyro_filtered_data_full_precision_2.xlsx")

# Time conversion
t = df["time"].to_numpy() * 1e-6
dt = np.median(np.diff(t))
fs = 1 / dt

print("Estimated sample rate:", fs, "Hz")


def classify_frequency(freq):
    if freq < 40:
        return "Flight motion or frame movement"
    elif 40 <= freq < 120:
        return "Frame resonance or loose structure"
    elif 120 <= freq < 300:
        return "Propeller imbalance"
    elif 300 <= freq < 600:
        return "Motor vibration or bearing issue"
    else:
        return "High-frequency electrical noise"


# Track strongest vibration
largest_peak_mag = 0
largest_peak_freq = 0
largest_axis = None
noise_level_global = 0

for axis in ["gyroADC[0]", "gyroADC[1]", "gyroADC[2]"]:

    x = df[axis].to_numpy()
    x = x - np.mean(x)

    N = len(x)

    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(fft_vals) / N

    # Ignore very low frequencies
    valid = freqs > 20

    # Estimate noise level (average magnitude)
    noise_level = np.mean(mag[valid])
    noise_level_global += noise_level

    # Find strongest peak
    idx = np.argmax(mag[valid])
    peak_mag = mag[valid][idx]
    peak_freq = freqs[valid][idx]

    if peak_mag > largest_peak_mag:
        largest_peak_mag = peak_mag
        largest_peak_freq = peak_freq
        largest_axis = axis

    # Plot (optional)
    plt.figure(figsize=(12,5))
    plt.plot(freqs, mag)
    plt.xlim(0,1000)
    plt.title(f"FFT of {axis}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

# Average noise across axes
noise_level_global /= 3

# Compute how strong the peak is relative to noise
ratio = largest_peak_mag / noise_level_global

# Health classification
if ratio < 5:
    status = "OK"
elif 5 <= ratio < 15:
    status = "WARNING"
else:
    status = "FAILURE"

problem = classify_frequency(largest_peak_freq)

print("\n=== Drone Diagnostic ===")
print(f"Strongest vibration: {largest_peak_freq:.1f} Hz ({largest_axis})")
print("Likely cause:", problem)
print("Health status:", status)