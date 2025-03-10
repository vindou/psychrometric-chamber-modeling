# IMPORTS
import numpy as np, pandas as pd, cvxpy as cp, matplotlib.pyplot as plt, control as ctrl, tqdm


def convert_F_to_K(T_F):
    return (T_F - 32) * 5/9 + 273.15

# GIVEN CONSTANTS
# The temperature of the adjacent chamber and the ambient environmnet
# is assumed to be constant at the moment, but in the future it will be
# a vector of temperatures that change over time.
dt = 1                    # TIME STEP [s]
C_air = 1.0012            # AIR SPECIFIC HEAT CAPACITY [kJ/(kg * K)]
T_amb = 77                # AMBIENT TEMPERATURE IN HIGH BAY [F]
T_2c = 95                 # ADJACENT CHAMBER SETPOINT [F]

T_amb = convert_F_to_K(T_amb) # CONVERT TO KELVIN
T_2c = convert_F_to_K(T_2c)   # CONVERT TO KELVIN

# LOAD IN DATA
data = pd.read_excel('Data_for_MATLAB.xlsx')

# ISOLATE COLUMNS OF INTEREST
T_thermostat = data['Room_Temperature'].to_numpy() # THERMOSTAT SETTING [F]
T_return = data['Return_Air_Temperature'].to_numpy() # RETURNING AIR TEMPERATURE [F]

Kp = 10
Ki = 0.0015

# control signal = Kp (error) + Ki(integral of error)

T_ctrl = np.zeros_like(T_thermostat)

integral = 0

for t in range(0, len(T_thermostat)):
    error = T_thermostat[t] - T_return[t]
    integral += error * dt
    T_ctrl[t] = Kp * error + Ki * integral

plt.figure(figsize=(10,6))
plt.plot(T_thermostat, label="Thermostat Setting")
plt.plot(T_return, label="Returning Air Temperature")
plt.plot(T_ctrl, label="Control Signal")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [F]")
plt.legend()
plt.grid(True)
plt.show()

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
print(f"UA_amb = {X_amb}")
print(f"UA_2c  = {X_2c}")

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

"""
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
"""