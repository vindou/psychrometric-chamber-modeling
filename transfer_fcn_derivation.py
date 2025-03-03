# IMPORTS
import numpy as np, pandas as pd, cvxpy as cp, matplotlib.pyplot as plt, control as ctrl, tqdm

# GIVEN CONSTANTS
# The temperature of the adjacent chamber and the ambient environmnet
# is assumed to be constant at the moment, but in the future it will be
# a vector of temperatures that change over time.
dt = 1                    # TIME STEP [s]
C_air = 1.005             # AIR SPECIFIC HEAT CAPACITY [kJ/(kg * K)]
T_amb = 77                # AMBIENT TEMPERATURE IN HIGH BAY [F]
T_2c = 95                 # ADJACENT CHAMBER SETPOINT [F]

# LOAD IN DATA
data = pd.read_excel('Data_for_MATLAB.xlsx')

# ISOLATE COLUMNS OF INTEREST
T_hc = data['Heat_Coil_Temperature'].to_numpy()    # HEAT COIL TEMPERATURE [F]
TC1 = data['Thermocouple_Top'].to_numpy()          # First thermocouple reading [F]
TC2 = data['Thermocouple_Bottom'].to_numpy()       # Second thermocouple reading [F]
T_r = (TC1 + TC2) / 2                              # AVERAGED ROOM TEMPERATURE [F]

# UNKNOWN PARAMETERS
Cap = 0;     # HEAT CAPACITY OF THE ROOM (NOT DIVIDED BY MASS FLOW OF AIR)
UA_amb = 0;  # HEAT TRANSFER COEFFICIENT BETWEEN THE ROOM AND THE AMBIENT ENVIRONMENT (Divided by mass flow of air)
UA_2c = 0;   # HEAT TRANSFER COEFFICIENT BETWEEN THE ROOM AND THE ADJACENT CHAMBER (Divided by mass flow of air)

# GOVERNING EQUATION: Cap * dT_r/dt = C_air * (T_hc - T_r) + UA_amb * (T_amb - T_r) + UA_2c * (T_2c - T_r)
# UA and C are divided by m_air to prevent unit mismatch.

dT_r = np.diff(T_r) / dt

A = np.column_stack([
    dT_r,                          # multiplies Cap
    -(T_amb - T_r[:-1]),           # multiplies UA_amb
    -(T_2c  - T_r[:-1])            # multiplies UA_2c
])
b = C_air * (T_hc[:-1] - T_r[:-1]) # right-hand side

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
Cap, UA_amb, UA_2c = x

print("Estimated parameters (Least Squares):")
print(f"Cap    = {Cap}")
print(f"UA_amb = {UA_amb}")
print(f"UA_2c  = {UA_2c}")

T_r_pred = np.zeros_like(T_r)
T_r_pred[0] = T_r[0] # ENFORCE INITIAL CONDITION

for t in tqdm.tqdm(range(len(T_r) - 1)):
    dTdt = (
        C_air * (T_hc[t] - T_r_pred[t])
        + UA_amb * (T_amb - T_r_pred[t])
        + UA_2c  * (T_2c  - T_r_pred[t])
    ) / Cap
    T_r_pred[t+1] = T_r_pred[t] + dt * dTdt

# --- Plot results ---
time = np.arange(len(T_r)) * dt

plt.figure(figsize=(10,6))
plt.plot(time, T_r,      label="Measured Room Temperature")
plt.plot(time, T_r_pred, label="Predicted Room Temperature", linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Room Temperature [°F]")
plt.title("Measured vs. Predicted Room Temperature")
plt.legend()
plt.grid(True)
plt.show()

# Define the transfer function
num = [C_air]  # numerator
den = [Cap, (C_air + UA_amb + UA_2c)]  # denominator
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