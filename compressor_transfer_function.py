import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import least_squares

# Load data
data = pd.read_excel('Simulation_Data.xlsx')

# Extract relevant signals
compressor_speed = data['Compressor Speed (rpm)'].values  # Compressor speed [RPM]
cooling_capacity = data['Air Side Cooling Capacity (kW)'].values  # Air-side cooling capacity [BTU/hr]

# Time vector (assuming 1 second sampling)
dt = 0.05  # sampling time [s]
t = np.arange(0, len(compressor_speed)*dt, dt)

# Remove any NaN or invalid values
valid_idx = ~np.isnan(compressor_speed) & ~np.isnan(cooling_capacity)
t = t[valid_idx]
compressor_speed = compressor_speed[valid_idx]
cooling_capacity = cooling_capacity[valid_idx]

# For a first-order system: τ*dy/dt + y = K*u
# where y is cooling_capacity and u is compressor_speed

# Calculate derivative of cooling capacity
dQdt = np.zeros_like(cooling_capacity)
dQdt[1:] = np.diff(cooling_capacity)/dt
dQdt[0] = dQdt[1]  # Use forward difference for first point

# Define the error function for least squares
def error_function(params):
    K, tau = params
    # τ*dQdt + Q = K*N
    return tau*dQdt + cooling_capacity - K*compressor_speed

# Initial guess for parameters
initial_guess = [1.0, 1.0]

# Solve for parameters using least squares
result = least_squares(error_function, initial_guess, bounds=([0, 0], [np.inf, np.inf]))
K, tau = result.x

# Generate predicted cooling capacity
Q_pred = np.zeros_like(cooling_capacity)
Q_pred[0] = cooling_capacity[0]  # Initial condition

for i in range(1, len(Q_pred)):
    dQdt = (K*compressor_speed[i-1] - Q_pred[i-1])/tau
    Q_pred[i] = Q_pred[i-1] + dt*dQdt

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, compressor_speed)
plt.grid(True)
plt.title('Compressor Speed')
plt.xlabel('Time [s]')
plt.ylabel('Speed [RPM]')

plt.subplot(3, 1, 2)
plt.plot(t, cooling_capacity, 'b', label='Measured')
plt.plot(t, Q_pred, 'r--', label='Predicted')
plt.grid(True)
plt.title('Cooling Capacity: Measured vs Predicted')
plt.xlabel('Time [s]')
plt.ylabel('Cooling Capacity [BTU/hr]')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, cooling_capacity - Q_pred)
plt.grid(True)
plt.title('Model Error')
plt.xlabel('Time [s]')
plt.ylabel('Error [BTU/hr]')

plt.tight_layout()
plt.show()

# Display identified parameters
print('\nIdentified First-Order System Parameters:')
print(f'Gain (K) = {K:.4f} BTU/hr/RPM')
print(f'Time Constant (τ) = {tau:.4f} s')

# Calculate fit quality
R2 = 1 - np.sum((cooling_capacity - Q_pred)**2)/np.sum((cooling_capacity - np.mean(cooling_capacity))**2)
print(f'R² = {R2:.4f}')

# Transfer function representation
print('\nTransfer Function:')
print(f'G(s) = {K:.4f}/({tau:.4f}s + 1)')

# Plot step response
t_step = np.linspace(0, 5*tau, 1000)  # Time vector for step response
t_step, y_step = signal.step(([K], [tau, 1]), T=t_step)

plt.figure()
plt.plot(t_step, y_step)
plt.grid(True)
plt.title('Step Response of Identified System')
plt.xlabel('Time [s]')
plt.ylabel('Cooling Capacity [BTU/hr]')
plt.show() 