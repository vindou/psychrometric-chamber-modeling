import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# add a PI controller
# take note of measurement block, thermocouples are not instantly responsive: sensor dynamics

# system possibly unstable across -inf to inf, but stable for bounded inputs (bibo stable?), control saturation possible nonlinearity
# block diagram

# Plant parameters
num = [1.005]
den = [4.938, 0.98]
G = ctrl.tf(num, den)
print("Plant G(s):", G)

# Desired closed-loop pole location
s_des = -100

K = (-4.938 * s_des - 0.98) / 1.005
print("Calculated proportional gain K:", K)

# Define the controller
D = ctrl.tf([K], [1])

# Closed-loop system with unity feedback
sys_cl = ctrl.feedback(D*G, 1)
print("Closed-loop transfer function:", ctrl.tf(sys_cl))

# Check the closed-loop pole
cl_poles = ctrl.poles(sys_cl)
print("Closed-loop poles:", cl_poles)

# Optional: step response to visualize dynamics
time, response = ctrl.step_response(sys_cl)
plt.plot(time, response)
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.title('Closed-loop Step Response')
plt.grid(True)
plt.show()
