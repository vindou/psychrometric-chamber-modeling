# IMPORTS
import numpy as np, pandas as pd, matplotlib.pyplot as plt, control as ctrl, tqdm


def convert_F_to_K(T_F):
    return (T_F - 32) * 5/9 + 273.15

# GIVEN CONSTANTS
dt = 1                    # TIME STEP [s]
C_air = 1.0012            # AIR SPECIFIC HEAT CAPACITY

# LOAD IN DATA
data = pd.read_excel('Data_for_MATLAB.xlsx')

# ISOLATE COLUMNS OF INTEREST
TC1 = data['Thermocouple_Top'].to_numpy()          # TOP THERMOCOUPLE READING [F]
TC2 = data['Thermocouple_Bottom'].to_numpy()       # BOTTOM THERMOCOUPLE READING [F]
T_r = (TC1 + TC2) / 2                              # AVERAGED ROOM TEMPERATURE [F]
T_return = data['Return_Air_Temperature'].to_numpy() # RETURNING AIR TEMPERATURE [F]

# Convert temperatures to Kelvin
T_r = convert_F_to_K(T_r)
T_return = convert_F_to_K(T_return)

# UNKNOWN PARAMETERS
Cap = 1.0;     # HEAT CAPACITY OF THE AIR IN THE DUCT (initial guess)
X = 0.1;       # HEAT TRANSFER COEFFICIENT (Divided by mass flow of air) (initial guess)

# Compute derivatives
dT_return = np.diff(T_return) / dt

# Set up least squares problem with regularization
A = np.column_stack([
    dT_return,           # multiplies Cap
    -(T_return[:-1] - T_r[:-1])  # multiplies X
])

# Add regularization to prevent zero values
reg_weight = 1e-6
A_reg = np.vstack([A, np.array([[reg_weight, 0], [0, reg_weight]])])
b_reg = np.concatenate([np.zeros_like(dT_return), np.array([reg_weight, reg_weight])])

x, residuals, rank, s = np.linalg.lstsq(A_reg, b_reg, rcond=None)
Cap, X = x

# Ensure parameters are positive
Cap = max(Cap, 1e-6)
X = max(X, 1e-6)

print("Estimated parameters (Least Squares):")
print(f"Cap = {Cap}")
print(f"X   = {X}")

# Generate predictions
T_return_pred = np.zeros_like(T_return)
T_return_pred[0] = T_return[0]  # Initial condition

for t in tqdm.tqdm(range(len(T_return) - 1)):
    dTdt = X * (T_r[t] - T_return_pred[t]) / Cap
    T_return_pred[t+1] = T_return_pred[t] + dt * dTdt

# --- Plot results ---
time = np.arange(len(T_return)) * dt

plt.figure(figsize=(10,6))
plt.plot(time, T_return, label="Measured Return Temperature")
plt.plot(time, T_return_pred, label="Predicted Return Temperature", linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Return Temperature [K]")
plt.title("Measured vs. Predicted Return Temperature")
plt.legend()
plt.grid(True)
plt.show()

# Define the transfer function
num = [X]  # numerator
den = [Cap, X]  # denominator
sys = ctrl.tf(num, den)
print("\nTransfer Function:", sys)

# Continuous-time stability analysis
poles = ctrl.poles(sys)
print("\nContinuous-time Poles:", poles)

plt.figure()
ctrl.pole_zero_plot(sys, plot=True)
plt.title("Continuous-time Pole-Zero Map")
plt.grid(True)
plt.show()

if np.all(np.real(poles) < 0):
    print("\nThe continuous-time system is stable.")
else:
    print("\nThe continuous-time system is unstable.")

# Z-domain stability analysis
sys_d = ctrl.c2d(sys, dt, method='zoh')
print("\nDiscrete-time Transfer Function:", sys_d)

z_poles = ctrl.poles(sys_d)
print("\nZ-domain Poles:", z_poles)

if np.all(np.abs(z_poles) < 1):
    print("\nThe discrete-time system is stable (all poles inside unit circle).")
else:
    print("\nThe discrete-time system is unstable (poles outside unit circle).")

plt.figure()
ctrl.pole_zero_plot(sys_d, plot=True)
plt.title("Z-Domain Pole-Zero Map")
plt.grid(True)
plt.show()

# Plot unit circle for reference
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
plt.plot(x, y, 'k--', alpha=0.5, label='Unit Circle')
plt.legend()
plt.axis('equal')
plt.show()