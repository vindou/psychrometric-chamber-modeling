import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
from scipy.optimize import least_squares

def main():
    df = pd.read_csv('heat_pump_model_data.csv')
    t   = df['Time (s)'].values - df['Time (s)'].iloc[0]
    rpm = df['Compressor Speed (rpm)'].values
    cap = df['Air Side Cooling Capacity (kW)'].values

    u = rpm
    y = cap

    K_dc = (y[-1] - y[0]) / (u[-1] - u[0])

    zeta0 = 0.01
    wn0   = 265.9
    a1_0  = 2*zeta0*wn0
    a0_0  = wn0**2

    b1_0 = 0.0
    b0_0 = K_dc * a0_0

    y0_0 = y[0] - (b0_0/a0_0)*u[0]

    p0 = [b1_0, b0_0, a1_0, a0_0, y0_0]
    bounds = ([0,    0,    0,    0,   -np.inf],
              [np.inf,np.inf,np.inf,np.inf, np.inf])

    def simulate(params):
        b1, b0, a1, a0, y0 = params
        sys = lti([b1, b0], [1, a1, a0])
        _, y_tf, _ = lsim(sys, U=u, T=t)
        return y_tf + y0

    def residual(params):
        return simulate(params) - y

    res = least_squares(residual, p0, bounds=bounds)
    b1, b0, a1, a0, y0 = res.x

    print("Fitted transfer function:")
    print(f"  G(s) = ({b1:.3e}s + {b0:.3e})" +
          f" / (s² + {a1:.3e}s + {a0:.3e})  +  {y0:.3e}")

    y_sim = simulate(res.x)
    plt.figure()
    plt.plot(t,    y,     label='Measured capacity (kW)')
    plt.plot(t,    y_sim, '--', label='Model response')
    plt.xlabel('Time (s)')
    plt.ylabel('Cooling capacity (kW)')
    plt.title('Raw RPM→Capacity Fit (2nd‑order+zero + offset)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
