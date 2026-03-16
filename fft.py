import pandas as pd          # Import pandas to read and manipulate spreadsheet data
import numpy as np           # Import numpy for numerical calculations and FFT
import matplotlib.pyplot as plt  # Import matplotlib to create graphs

# Read the Excel file containing the gyro data
df = pd.read_excel("gyro_filtered_data_full_precision_2.xlsx")

# Convert the time column to seconds
# The original data is likely in microseconds, so multiply by 10^-6
t = df["time"].to_numpy() * 1e-6

# Compute the typical time step between measurements
# np.diff finds differences between consecutive time values
dt = np.median(np.diff(t))

# Compute the sampling frequency of the sensor
# sampling rate = 1 / time between samples
fs = 1 / dt

# Print the estimated sampling rate to the user
print("Estimated sample rate:", fs, "Hz")


# Function that classifies a vibration frequency into a possible drone problem
def classify_frequency(freq):

    # Very low frequencies usually come from drone motion or pilot inputs
    if freq < 40:
        return "Flight motion or frame movement"

    # Moderate frequencies can indicate structural vibration
    elif 40 <= freq < 120:
        return "Frame resonance or loose structure"

    # These frequencies often correspond to propeller imbalance
    elif 120 <= freq < 300:
        return "Propeller imbalance"

    # Higher frequencies are typically caused by motor vibration
    elif 300 <= freq < 600:
        return "Motor vibration or bearing issue"

    # Extremely high frequencies are usually electrical noise
    else:
        return "High-frequency electrical noise"


# Variables to track the strongest vibration detected
largest_peak_mag = 0      # Stores the magnitude of the strongest peak found
largest_peak_freq = 0     # Stores the frequency of the strongest peak
largest_axis = None       # Stores which gyro axis produced that peak


# Loop through the three gyro axes
for axis in ["gyroADC[0]", "gyroADC[1]", "gyroADC[2]"]:

    # Extract the gyro signal for the current axis
    x = df[axis].to_numpy()

    # Remove the average value from the signal
    # This eliminates DC offset so the FFT focuses on vibrations
    x = x - np.mean(x)

    # Determine the number of samples in the signal
    N = len(x)

    # Compute the FFT of the signal
    # rfft is used because the signal is real-valued and we only need positive frequencies
    fft_vals = np.fft.rfft(x)

    # Generate the frequency values corresponding to the FFT output
    freqs = np.fft.rfftfreq(N, d=dt)

    # Compute the magnitude (strength) of each frequency component
    mag = np.abs(fft_vals) / N

    # Ignore very low frequencies (below 20 Hz)
    # These usually correspond to drone movement, not mechanical vibration
    valid = freqs > 20

    # Find the index of the largest magnitude within the valid frequencies
    idx = np.argmax(mag[valid])

    # Get the magnitude of that peak
    peak_mag = mag[valid][idx]

    # Get the frequency where that peak occurs
    peak_freq = freqs[valid][idx]

    # If this peak is larger than the previous largest peak, store it
    if peak_mag > largest_peak_mag:
        largest_peak_mag = peak_mag
        largest_peak_freq = peak_freq
        largest_axis = axis

    # Plot the FFT spectrum for the current axis
    plt.figure(figsize=(12,5))

    # Plot frequency vs magnitude
    plt.plot(freqs, mag)

    # Limit the graph to 0–1000 Hz for easier interpretation
    plt.xlim(0,1000)

    # Title indicating which axis is being shown
    plt.title(f"FFT of {axis}")

    # Label the x-axis
    plt.xlabel("Frequency (Hz)")

    # Label the y-axis
    plt.ylabel("Magnitude")

    # Display grid lines for easier reading
    plt.grid(True)

    # Show the graph
    plt.show()


# After analyzing all axes, classify the strongest vibration frequency
problem = classify_frequency(largest_peak_freq)

# Print the final diagnostic result
print("\nMost likely issue detected:")

# Report where the strongest vibration occurred
print(f"Strongest vibration at {largest_peak_freq:.1f} Hz on {largest_axis}")

# Output the most likely cause of that vibration
print("Diagnosis:", problem)