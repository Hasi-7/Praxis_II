import serial
import time
import numpy as np

# Match these to your analyze.py settings
NUM_SAMPLES = 2048
SAMPLE_RATE = 20000
FUNDAMENTAL_FREQ = 1332
FAULT_FREQ = 3730

# Change to whichever virtual port is the "Arduino" end
VIRTUAL_PORT = 'COM10'  # Mac/Linux: '/dev/pts/2'

def generate_signal(mode='healthy', noise_pct=0.0):
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * FUNDAMENTAL_FREQ * t)
    if mode == 'faulty':
        signal += 0.005 * np.sin(2 * np.pi * FAULT_FREQ * t)
    if noise_pct > 0:
        signal += np.random.normal(0, 0.5 * noise_pct, NUM_SAMPLES)
    # Shift to 0-5V range and convert to 10-bit ADC integers
    signal += 0.5
    signal = np.clip(signal, 0.0, 5.0)
    return (signal / 5.0 * 1023).astype(int)

print(f"Fake Arduino running on {VIRTUAL_PORT}")
print("Modes: type 1=healthy, 2=faulty clean, 3=faulty+5% noise, 4=faulty+20% noise")

ser = serial.Serial(VIRTUAL_PORT, 115200, timeout=5)
time.sleep(1)
ser.write(b'READY\n')

while True:
    # Choose which signal to emit — change this line to test different modes
    mode = 'faulty'
    noise = 0.05

    if ser.read(1) == b'G':
        print(f"Triggered — sending {NUM_SAMPLES} samples ({mode}, noise={noise*100:.0f}%)")
        samples = generate_signal(mode=mode, noise_pct=noise)
        ser.write(b'START\n')
        for s in samples:
            ser.write(f"{s}\n".encode())
        ser.write(b'END\n')
        print("Done sending")