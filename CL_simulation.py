"""
CL_simulation.py
Closed-loop simulation: PI controller + heater-plant

Reference (r):   Low_Coil_Temp_1107  ≈ 90 °F
Measurement (y): Room_T_1107         (°F)
Controller out:  Heat_Coil_Temp_1107 (°F)

PI:    Kp = 10, Ki = 0.0015   (parallel form)
Plant: P(s) = 1.0012 / (15.86 s + 1.322)

Requires SciPy ≥ 1.12 (lsim needs 2-D U, uniform T)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ───────────── 1. Load CSV ─────────────
CSV = "CL_test.csv"
df  = pd.read_csv(CSV)

time_raw = pd.to_datetime(df["DateTime"])
t        = (time_raw - time_raw.iloc[0]).dt.total_seconds().values        # [s]

r_F = df["Low_Coil_Temp_1107"].values      # set-point, ~90 °F
y_F = df["Room_T_1107"].values             # measured room temp
u_F = df["Heat_Coil_Temp_1107"].values     # measured PI output (coil cmd)

# °F ↔ K helpers
f2k = lambda F: (5/9)*(F - 32) + 273.15
k2f = lambda K: (K - 273.15)*9/5 + 32

r_K, y_K = f2k(r_F), f2k(y_F)
u0_K     = f2k(u_F[0])                     # initial coil temp (baseline)

# ───────────── 2. Controller & Plant ─────────────
Kp, Ki = 10.0, 0.0015
C_num, C_den = [Kp, Ki], [1, 0]            # (Kp s + Ki)/s
P_num, P_den = [1.0012], [15.86, 1.322]

def tf_mul(num1, den1, num2, den2):
    return np.polymul(num1, num2), np.polymul(den1, den2)

def tf_add1(num, den):
    """Return numerator, denominator of 1 + num/den."""
    return np.polyadd(den, num), den

# Open-loop L(s) = C·P
L_num, L_den = tf_mul(C_num, C_den, P_num, P_den)

# 1 + L
sum_num, _ = tf_add1(L_num, L_den)         # numerator of (1+L)

# T_ry(s) = L / (1+L)         r → y
T_ry = signal.TransferFunction(L_num, sum_num)

# T_ru(s) = C / (1+L)         r → u
T_ru_num, _ = tf_mul(C_num, C_den, L_den, [1])   # C_num * L_den
T_ru_den = np.polymul(C_den, sum_num)            # C_den * (L_num+L_den)
T_ru = signal.TransferFunction(T_ru_num, T_ru_den)

# ───────────── 3. Uniform grid & interpolation ─────────────
dt      = np.median(np.diff(t))                # robust step (≈1 s)
t_uni   = np.arange(0, t[-1] + dt, dt)

r_dev = r_K - r_K[0]                           # deviation input
r_dev_uni    = np.interp(t_uni, t, r_dev)
y_meas_F_uni = np.interp(t_uni, t, y_F)
u_meas_F_uni = np.interp(t_uni, t, u_F)

U_mat = r_dev_uni[:, None]                     # (N,1) for lsim

# ───────────── 4. Closed-loop simulation ─────────────
_, y_dev_sim, _ = signal.lsim(T_ry, U=U_mat, T=t_uni)
_, u_dev_sim, _ = signal.lsim(T_ru, U=U_mat, T=t_uni)

# Add biases back
y_sim_K = y_dev_sim + r_K[0]
u_sim_K = u_dev_sim + u0_K                    # align coil-cmd baseline

y_sim_F = k2f(y_sim_K)
u_sim_F = k2f(u_sim_K)

# ───────────── 5. Plots ─────────────
plt.figure(figsize=(11, 5))

plt.subplot(2, 1, 1)
plt.plot(t_uni, y_meas_F_uni, label="Measured Room_T_1107")
plt.plot(t_uni, y_sim_F, "--", label="Simulated Room Temp")
plt.ylabel("Room Temp (°F)")
plt.legend(); plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_uni, u_meas_F_uni, label="Measured Heat_Coil_Temp_1107")
plt.plot(t_uni, u_sim_F, "--", label="Simulated PI Output")
plt.xlabel("Time (s)")
plt.ylabel("Heater Coil Cmd (°F)")
plt.legend(); plt.grid(True)

plt.suptitle("Closed-Loop Simulation  •  PI(Kp=10, Ki=0.0015)  +  P(s)")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
