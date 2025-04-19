import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('heat_pump_data.csv')
df = df[['Compressor Speed (rpm)', 'Air Side Cooling Capacity (kW)', 'Time (s)']]
df = df.iloc[1999:]
df.to_csv('heat_pump_model_data.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(df['Time (s)'], df['Air Side Cooling Capacity (kW)'])
plt.xlabel('Time (s)')
plt.ylabel('Air Side Cooling Capacity (kW)')
plt.title('Air Side Cooling Capacity vs Time')
plt.grid(True)
plt.show()
plt.close()