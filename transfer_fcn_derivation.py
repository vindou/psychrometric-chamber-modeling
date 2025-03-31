# IMPORTS
import numpy as np, pandas as pd, cvxpy as cp, matplotlib.pyplot as plt, control as ctrl, tqdm


def convert_F_to_K(T_F):
    return (T_F - 32) * 5/9 + 273.15

# GIVEN CONSTANTS
# The temperature of the adjacent chamber and the ambient environmnet
# is assumed to be constant at the moment, but in the future it will be
# a vector of temperatures that change over time.
dt = 1                    # TIME STEP [s]
C_air = 1.0012            # AIR SPECIFIC HEAT CAPACITY
T_amb = 77                # AMBIENT TEMPERATURE IN HIGH BAY [F]
T_2c = 95                 # ADJACENT CHAMBER SETPOINT [F]

T_amb = convert_F_to_K(T_amb) # CONVERT TO KELVIN
T_2c = convert_F_to_K(T_2c)   # CONVERT TO KELVIN

# LOAD IN DATA
data = pd.read_excel('Data_for_MATLAB.xlsx')

# ISOLATE COLUMNS OF INTEREST
T_thermostat = data['Room_Temperature'].to_numpy() # THERMOSTAT SETTING [F]
T_return = data['Return_Air_Temperature'].to_numpy() # RETURNING AIR TEMPERATURE [F]

"""
The heat pump's PI controller has the gains Kp = 10 and Ki = 0.0015;
the control signal is 0 to 100% of the heat pump's maximum power. In
the given data, the heat pump's control signal was overrided by the
researchers overseeing the test, so closed loop poles cannot be derived.
"""

T_hc = data['Heat_Coil_Temperature'].to_numpy()    # HEAT COIL TEMPERATURE [F]

TC1 = data['Thermocouple_Top'].to_numpy()          # TOP THERMOCOUPLE READING [F]
TC2 = data['Thermocouple_Bottom'].to_numpy()       # BOTTOM THERMOCOUPLE READING [F]
T_r = (TC1 + TC2) / 2                              # AVERAGED ROOM TEMPERATURE [F]
T_r = convert_F_to_K(T_r)                          # CONVERT TO KELVIN
T_hc = convert_F_to_K(T_hc)                        # CONVERT TO KELVIN

# UNKNOWN PARAMETERS
Cap = 0;     # HEAT CAPACITY OF THE ROOM (NOT DIVIDED BY MASS FLOW OF AIR)
X_amb = 0;  # HEAT TRANSFER COEFFICIENT BETWEEN THE ROOM AND THE AMBIENT ENVIRONMENT (Divided by mass flow of air)
X_2c = 0;   # HEAT TRANSFER COEFFICIENT BETWEEN THE ROOM AND THE ADJACENT CHAMBER (Divided by mass flow of air)

dT_r = np.diff(T_r) / dt

A = np.column_stack([
    dT_r,                          # multiplies Cap
    -(T_r[:-1] - T_amb),           # multiplies UA_amb
    -(T_2c  - T_r[:-1])            # multiplies UA_2c
])
b = C_air * (T_hc[:-1] - T_r[:-1]) # right-hand side

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
Cap, X_amb, X_2c = x

print("Estimated parameters (Least Squares):")
print(f"Cap    = {Cap}")
print(f"X_amb = {X_amb}")
print(f"X_2c  = {X_2c}")

T_r_pred = np.zeros_like(T_r)
T_r_pred[0] = T_r[0] # ENFORCE INITIAL CONDITION

for t in tqdm.tqdm(range(len(T_r) - 1)):
    dTdt = (
        C_air * (T_hc[t] - T_r_pred[t])
        + X_amb * (T_r_pred[t] - T_amb)
        + X_2c  * (T_2c  - T_r_pred[t])
    ) / Cap
    T_r_pred[t+1] = T_r_pred[t] + dt * dTdt

# --- Plot results ---
time = np.arange(len(T_r)) * dt

plt.figure(figsize=(10,6))
plt.plot(time, T_r,      label="Measured Room Temperature")
plt.plot(time, T_r_pred, label="Predicted Room Temperature", linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Room Temperature [K]")
plt.title("Measured vs. Predicted Room Temperature")
plt.legend()
plt.grid(True)
plt.show()

# Define the transfer function
num = [C_air]  # numerator
den = [Cap, (C_air + X_amb + X_2c)]  # denominator
sys = ctrl.tf(num, den)
print("Transfer Function:", sys)

poles = ctrl.poles(sys)
print("Poles:", poles)

ctrl.pole_zero_plot(sys, plot=True)
plt.title("Pole-Zero Map")
plt.show()

if np.all(np.real(poles) < 0):
    print("The system is stable.")
else:
    print("The system is unstable.")

# --- Z-Domain Stability Analysis ---
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