import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
from scipy.optimize import least_squares

df = pd.read_csv('heat_pump_model_data.csv')
t = df['Time (s)'].values - df['Time (s)'].iloc[0]
rpm = df['Compressor Speed (rpm)'].values
cap = df['Air Side Cooling Capacity (kW)'].values

u = rpm - rpm[0]
y = cap - cap[0]

def simulate(params):
    b1, b0, a1, a0 = params
    sys = lti([b1, b0], [1, a1, a0])
    _, y_sim, _ = lsim(sys, U=u, T=t)
    return y_sim

def resid(params):
    return simulate(params) - y

K_prev = y[-1]/u[-1]
ζ_prev, ωn_prev = 0.01, 265.9
b1_0 = 0.0
b0_0 = K_prev*ωn_prev**2
a1_0 = 2*ζ_prev*ωn_prev
a0_0 = ωn_prev**2
p0 = [b1_0, b0_0, a1_0, a0_0]

bounds = ([0, 0,   0,      0],
          [np.inf, np.inf, np.inf, np.inf])

res = least_squares(resid, p0, bounds=bounds)
b1, b0, a1, a0 = res.x

print("Fitted transfer function:")
print(f"  G(s) = ({b1:.3f}·s + {b0:.3f}) / (s² + {a1:.3f}·s + {a0:.3f})")

y_sim = simulate(res.x)
plt.figure()
plt.plot(t, y,      label='Measured ΔCapacity')
plt.plot(t, y_sim, '--', label='Fitted 2nd‑Order+Zero')
plt.xlabel('Time (s)')
plt.ylabel('Δ Cooling Capacity (kW)')
plt.legend()
plt.title('Step Response with Zero → captures initial overshoot')
plt.show()
