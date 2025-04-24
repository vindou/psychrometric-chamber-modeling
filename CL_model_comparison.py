import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

csv_file = "CL_test.csv"
df = pd.read_csv(csv_file)

t_raw = pd.to_datetime(df["DateTime"])
u_F   = df["Heat_Coil_Temp_1107"].values        # coil temp (F)
y_F   = df["Room_T_1107"].values                # room temp (F)

t = (t_raw - t_raw.iloc[0]).dt.total_seconds().values

f2k = lambda T_F: (5/9)*(T_F - 32) + 273.15
k2f = lambda T_K: (T_K - 273.15)*9/5 + 32

u_K = f2k(u_F)
y_K = f2k(y_F)

u_dev = u_K - u_K[0]
y_dev = y_K - y_K[0]

t_uni = np.linspace(t[0], t[-1], len(t))
u_dev_uni = np.interp(t_uni, t, u_dev)

num, den = [1.0012], [15.86, 1.322]
sys = signal.TransferFunction(num, den)

t_out, y_sim_dev_K, _ = signal.lsim(sys, U=u_dev_uni, T=t_uni)

y_sim_K = y_sim_dev_K + y_K[0]
y_sim_F = k2f(y_sim_K)
y_meas_F = np.interp(t_uni, t, y_F)

rmse = np.sqrt(np.mean((y_meas_F - y_sim_F)**2))
print(f"RMSE = {rmse:.2f} °F")

plt.figure(figsize=(10,5))
plt.plot(t_uni, y_meas_F,       label="Actual Room T (1107)")
plt.plot(t_out, y_sim_F, "--", label="Predicted Room Temperature")
plt.xlabel("Time  [s]"); plt.ylabel("Temperature  [F]")
plt.title("Predicted vs. Actual Room Temperature")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


