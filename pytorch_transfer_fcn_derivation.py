import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import control as ctrl


def convert_F_to_K(T_F):
    return (T_F - 32) * 5/9 + 273.15

# GIVEN CONSTANTS
dt = 1                    # time step [s]
C_air = 1.0012             # air heat capacity coefficient
T_amb = 77.0              # ambient temperature [F]
T_2c = 95.0               # adjacent chamber setpoint [F]

# LOAD DATA
data = pd.read_excel('Data_for_MATLAB.xlsx')
T_hc_np = data['Heat_Coil_Temperature'].to_numpy()  # heat coil temperature [F]
TC1 = data['Thermocouple_Top'].to_numpy()          # First thermocouple reading [F]
TC2 = data['Thermocouple_Bottom'].to_numpy()       # Second thermocouple reading [F]
T_r_np = (TC1 + TC2) / 2                              # AVERAGED ROOM TEMPERATURE [F]

T_r_np = convert_F_to_K(T_r_np)  # CONVERT TO KELVIN
T_hc_np = convert_F_to_K(T_hc_np)  # CONVERT TO KELVIN
T_amb = convert_F_to_K(T_amb)  # CONVERT TO KELVIN
T_2c = convert_F_to_K(T_2c)  # CONVERT TO KELVIN

# CONVERT DATA TO TORCH TENSORS
T_hc = torch.tensor(T_hc_np, dtype=torch.float32)
T_r = torch.tensor(T_r_np, dtype=torch.float32)

# COMPUTE TIME DERIVATIVE OF ROOM TEMPERATURE USING FORWARD FINITE DIFFERENCES
dT_r_dt = torch.zeros_like(T_r)
dT_r_dt[:-1] = (T_r[1:] - T_r[:-1]) / dt
dT_r_dt[-1] = dT_r_dt[-2]  # replicate the last value

# DEFINE THE MODEL WITH UNKNOWN PARAMETERS
class ParameterEstimator(nn.Module):
    def __init__(self):
        super(ParameterEstimator, self).__init__()
        # Initialize unknown parameters with a guess
        self.Cap = nn.Parameter(torch.tensor(1.0))
        self.UA_amb = nn.Parameter(torch.tensor(1.0))
        self.UA_2c = nn.Parameter(torch.tensor(1.0))

    def forward(self, T_hc, T_r, dT_r_dt):
        # Governing equation residual:
        # Cap*dT_r/dt = C_air*(T_hc - T_r) + UA_amb*(T_amb - T_r) + UA_2c*(T_2c - T_r)
        # Rearranged to residual = 0:
        residual = self.Cap * dT_r_dt - C_air*(T_hc - T_r) \
                   - self.UA_amb*(T_r - T_amb) - self.UA_2c*(T_2c - T_r)
        return residual

# INSTANTIATE THE MODEL
model = ParameterEstimator()

# DEFINE THE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TRAINING LOOP
num_epochs = 50000
print("Training...")
for epoch in tqdm.tqdm(range(num_epochs)):
    optimizer.zero_grad()
    residual = model(T_hc, T_r, dT_r_dt)
    loss = torch.mean(residual**2)  # Mean squared error of the residual
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Cap = {model.Cap.item():.4f}, UA_amb = {model.UA_amb.item():.4f}, UA_2c = {model.UA_2c.item():.4f}")

print("\nOptimized Parameters:")
print(f"Cap: {model.Cap.item():.4f}")
print(f"UA_amb: {model.UA_amb.item():.4f}")
print(f"UA_2c: {model.UA_2c.item():.4f}")

# Generate predictions using the optimized parameters
T_r_pred = torch.zeros_like(T_r)
T_r_pred[0] = T_r[0]  # Initial condition

for t in tqdm.tqdm(range(len(T_r) - 1)):
    dTdt = (
        C_air * (T_hc[t] - T_r_pred[t])
        + model.UA_amb * (T_r_pred[t] - T_amb)
        + model.UA_2c * (T_2c - T_r_pred[t])
    ) / model.Cap
    T_r_pred[t+1] = T_r_pred[t] + dt * dTdt

# Plot results
time = torch.arange(len(T_r)) * dt
plt.figure(figsize=(10,6))
plt.plot(time.numpy(), T_r.numpy(), label="Measured Room Temperature")
plt.plot(time.numpy(), T_r_pred.detach().numpy(), label="Predicted Room Temperature", linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Room Temperature [K]")
plt.title("Measured vs. Predicted Room Temperature")
plt.legend()
plt.grid(True)
plt.show()

# Define the transfer function
num = [C_air]  # numerator
den = [model.Cap.item(), (C_air + model.UA_amb.item() + model.UA_2c.item())]  # denominator
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
