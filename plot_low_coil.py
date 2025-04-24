import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('CL_test.csv')

# Convert DateTime column to datetime type
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['DateTime'], df['Low_Coil_Temp_1107'], 'r-', label='Low Coil Temperature')

# Customize the plot
plt.title('Low Coil Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°F)')
plt.grid(True)
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show() 