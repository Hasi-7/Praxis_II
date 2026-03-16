import pandas as pd
import matplotlib.pyplot as plt

# load the spreadsheet
df = pd.read_excel("gyro_filtered_data_full_precision_2.xlsx")

# convert time to seconds (Betaflight blackbox time is usually microseconds)
time = df["time"] * 1e-6

# gyro axes
gyro_x = df["gyroADC[0]"]
gyro_y = df["gyroADC[1]"]
gyro_z = df["gyroADC[2]"]

# plot X axis
plt.figure()
plt.plot(time, gyro_x)
plt.xlabel("Time (s)")
plt.ylabel("GyroADC[0]")
plt.title("Gyro X vs Time")
plt.show()

# plot Y axis
plt.figure()
plt.plot(time, gyro_y)
plt.xlabel("Time (s)")
plt.ylabel("GyroADC[1]")
plt.title("Gyro Y vs Time")
plt.show()

# plot Z axis
plt.figure()
plt.plot(time, gyro_z)
plt.xlabel("Time (s)")
plt.ylabel("GyroADC[2]")
plt.title("Gyro Z vs Time")
plt.show()