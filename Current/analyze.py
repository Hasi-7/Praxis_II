import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# --- Configuration ---
SERIAL_PORT = (
    "COM6"  # Change to your Arduino's port (Windows: COM6, Mac/Linux: /dev/ttyUSB0)
)
BAUD_RATE = 115200
NUM_SAMPLES = 2048
SAMPLE_RATE = 10000  # Default; can be overridden with --sample-rate
FUNDAMENTAL_FREQ = 1332  # Hz
FAULT_FREQ = 3730  # Hz
SNR_PASS_THRESHOLD = 3.0  # Minimum SNR to pass Req 1.2


def capture_from_arduino(port):
    print(f"Connecting to Arduino on {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=10)
    time.sleep(2)  # Wait for Arduino reset after serial connect

    # Wait for ready signal
    ser.readline()

    # Send trigger and start timer (Req 2.2 clock starts here)
    t_start = time.time()
    ser.write(b"G")
    print("Triggered capture...")

    # Wait for START marker
    wait_start_deadline = time.time() + 15
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "START":
            break
        if time.time() > wait_start_deadline:
            ser.close()
            raise TimeoutError(
                "Timed out waiting for Arduino START marker. Check that the uploaded sketch sample rate is realistic and matches --sample-rate."
            )

    # Read samples
    raw_samples = []
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "END":
            break
        raw_samples.append(int(line))

    ser.close()
    print(f"Captured {len(raw_samples)} samples")
    return np.array(raw_samples), t_start


def convert_to_voltage(raw_samples, adc_bits=10, vref=5.0):
    # Arduino Mega ADC: 10-bit, 0-5V reference
    return (raw_samples / (2**adc_bits - 1)) * vref


def analyze(voltage_samples, sample_rate, sensor_name, t_start, mode="faulty"):
    N = len(voltage_samples)

    if FUNDAMENTAL_FREQ >= sample_rate / 2:
        raise ValueError(
            f"Fundamental frequency {FUNDAMENTAL_FREQ} Hz is above the Nyquist limit for sample rate {sample_rate} Hz"
        )

    fault_in_range = FAULT_FREQ < sample_rate / 2
    if not fault_in_range:
        print(
            f"Warning: fault frequency {FAULT_FREQ} Hz is above the Nyquist limit for sample rate {sample_rate} Hz and cannot be measured reliably."
        )

    # Remove DC offset
    voltage_samples = voltage_samples - np.mean(voltage_samples)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(N)
    windowed = voltage_samples * window

    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(windowed)) * 2 / N
    freqs = np.fft.rfftfreq(N, d=1 / sample_rate)

    # Find amplitude at fundamental and fault frequencies
    def peak_near(target_freq, tolerance=50):
        mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
        if not np.any(mask):
            return 0.0
        idx = np.argmax(fft_vals[mask])
        return fft_vals[mask][idx]

    fund_amp = peak_near(FUNDAMENTAL_FREQ)
    fault_amp = peak_near(FAULT_FREQ) if fault_in_range else 0.0

    # Compute noise floor (RMS of FFT excluding signal peaks)
    noise_mask = np.ones(len(freqs), dtype=bool)
    for f in [FUNDAMENTAL_FREQ, FAULT_FREQ]:
        noise_mask &= ~((freqs >= f - 100) & (freqs <= f + 100))
    noise_floor_rms = np.sqrt(np.mean(fft_vals[noise_mask] ** 2))

    # SNR calculation
    fault_snr = fault_amp / noise_floor_rms if noise_floor_rms > 0 else 0
    fundamental_snr = fund_amp / noise_floor_rms if noise_floor_rms > 0 else 0

    if mode == "healthy":
        metric_name = "Fundamental/noise"
        metric_value = fundamental_snr
    else:
        metric_name = "Fault/noise"
        metric_value = fault_snr

    # Stop timer (Req 2.2 clock ends here)
    t_end = time.time()
    elapsed = t_end - t_start

    # Results
    print(f"\n--- Results for {sensor_name} ---")
    print(f"Sample rate:                               {sample_rate} Hz")
    print(f"Fundamental amplitude @ {FUNDAMENTAL_FREQ} Hz: {fund_amp * 1000:.2f} mV")
    print(f"Fault amplitude      @ {FAULT_FREQ} Hz:  {fault_amp * 1000:.2f} mV")
    print(f"Noise floor RMS:                          {noise_floor_rms * 1000:.2f} mV")
    print(f"SNR (fundamental/noise):                 {fundamental_snr:.2f}")
    print(f"SNR (fault/noise):                       {fault_snr:.2f}")
    print(f"Time to result:                           {elapsed:.2f} seconds")
    print(
        f"\nReq 1.2 verdict ({metric_name}):  {'PASS' if metric_value >= SNR_PASS_THRESHOLD else 'FAIL'} (threshold: SNR >= {SNR_PASS_THRESHOLD})"
    )
    print(
        f"Req 2.2 verdict:  {'PASS' if elapsed <= 15 else 'FAIL'} (threshold: <= 15 seconds)"
    )

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, fft_vals * 1000, color="steelblue", linewidth=0.8)
    plt.axvline(
        FUNDAMENTAL_FREQ,
        color="green",
        linestyle="--",
        label=f"Fundamental ({FUNDAMENTAL_FREQ} Hz)",
    )
    plt.axvline(
        FAULT_FREQ, color="red", linestyle="--", label=f"Fault ({FAULT_FREQ} Hz)"
    )
    plt.axhline(
        noise_floor_rms * 1000,
        color="orange",
        linestyle=":",
        label=f"Noise floor ({noise_floor_rms * 1000:.2f} mV RMS)",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (mV)")
    plt.title(
        f"FFT Spectrum — {sensor_name} | {metric_name} SNR: {metric_value:.1f} | {'PASS' if metric_value >= SNR_PASS_THRESHOLD else 'FAIL'}"
    )
    plt.legend()
    plt.xlim(0, sample_rate / 2)
    plt.tight_layout()
    filename = f"fft_{sensor_name}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved as {filename}")


def generate_synthetic_samples(
    mode="healthy", noise_pct=0.0, num_samples=2048, sample_rate=20000
):
    """
    Generate synthetic voltage samples mimicking Arduino capture.

    mode options:
        'healthy'  - fundamental only (1332 Hz)
        'faulty'   - fundamental + fault sideband (3730 Hz at 1% amplitude)

    noise_pct: noise amplitude as percentage of fundamental (0.0 = no noise,
               0.05 = 5%, 0.10 = 10%, 0.20 = 20%)
    """
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)

    # Fundamental: 1332 Hz, 0.5V amplitude (representing 1.0Vpp centered at 0.5V)
    fundamental_amp = 0.5
    signal = fundamental_amp * np.sin(2 * np.pi * FUNDAMENTAL_FREQ * t)

    if mode == "faulty":
        # Fault sideband: 3730 Hz at 1% of fundamental
        fault_amp = fundamental_amp * 0.01
        signal += fault_amp * np.sin(2 * np.pi * FAULT_FREQ * t)

    if noise_pct > 0.0:
        # Broadband white Gaussian noise
        noise_amp = fundamental_amp * noise_pct
        noise = np.random.normal(0, noise_amp, num_samples)
        signal += noise

    # Add DC offset (0.5V) and clip to 0-5V to mimic Arduino ADC input range
    signal += 0.5
    signal = np.clip(signal, 0.0, 5.0)

    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor", required=True, help="Sensor label (e.g. ACS712, INA240, INA180)"
    )
    parser.add_argument("--port", default=SERIAL_PORT, help="Arduino serial port")
    parser.add_argument("--load", help="Load previously saved .npy samples file")
    parser.add_argument(
        "--synthetic",
        choices=["healthy", "faulty"],
        help="Use synthetic data instead of Arduino (for testing)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise level as decimal fraction e.g. 0.05 for 5%%",
    )
    parser.add_argument(
        "--mode",
        choices=["healthy", "faulty"],
        default="faulty",
        help="Use healthy mode to score the fundamental peak; use faulty mode to score the fault peak",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="Sampling rate in Hz; must match the Arduino sketch",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.synthetic:
        # --- SYNTHETIC MODE: no hardware needed ---
        print(
            f"[SYNTHETIC MODE] Generating {args.synthetic} signal "
            f"with {args.noise * 100:.0f}% noise..."
        )
        voltage_samples = generate_synthetic_samples(
            mode=args.synthetic,
            noise_pct=args.noise,
            num_samples=NUM_SAMPLES,
            sample_rate=args.sample_rate,
        )
        print(f"Generated {len(voltage_samples)} samples")

    elif args.load:
        # --- OFFLINE MODE: re-analyze previously saved capture ---
        voltage_samples = np.load(args.load)
        print(f"Loaded {len(voltage_samples)} samples from {args.load}")

    else:
        # --- LIVE MODE: capture from Arduino ---
        raw, t_start = capture_from_arduino(args.port)
        voltage_samples = convert_to_voltage(raw)
        np.save(f"samples_{args.sensor}.npy", voltage_samples)
        print(f"Samples saved to samples_{args.sensor}.npy")

    analyze(voltage_samples, args.sample_rate, args.sensor, t_start, mode=args.mode)
