% Load data
data = readtable('Simulation_Data.xlsx');

% Extract relevant signals
T_setpoint = data.Room_Temperature;  % Setpoint [F]
T_measured = (data.Thermocouple_Top + data.Thermocouple_Bottom) / 2;  % Measured temp [F]
duty_cycle = data.Duty_Cycle;  % Controller output [%]

% Calculate error signal
error = T_setpoint - T_measured;

% Time vector (assuming 1 second sampling)
dt = 1;  % sampling time [s]
t = (0:dt:(length(error)-1)*dt)';

% Remove any NaN or invalid values
valid_idx = ~isnan(error) & ~isnan(duty_cycle);
t = t(valid_idx);
error = error(valid_idx);
duty_cycle = duty_cycle(valid_idx);

% For a PI controller: u(t) = Kp*e(t) + Ki*∫e(t)dt
% where u is duty_cycle and e is error

% Calculate integral of error
error_integral = cumtrapz(t, error);

% Set up least squares problem to find Kp and Ki
% duty_cycle = Kp*error + Ki*error_integral
A = [error, error_integral];
b = duty_cycle;

% Solve for parameters
params = A\b;
Kp = params(1);
Ki = params(2);

% Generate predicted duty cycle using identified parameters
duty_cycle_pred = Kp*error + Ki*error_integral;

% Plot results
figure;
subplot(3,1,1);
plot(t, error);
grid on;
title('Error Signal');
xlabel('Time [s]');
ylabel('Error [°F]');

subplot(3,1,2);
plot(t, duty_cycle, 'b', t, duty_cycle_pred, 'r--');
grid on;
title('Duty Cycle: Measured vs Predicted');
xlabel('Time [s]');
ylabel('Duty Cycle [%]');
legend('Measured', 'Predicted');

subplot(3,1,3);
plot(t, duty_cycle - duty_cycle_pred);
grid on;
title('Model Error');
xlabel('Time [s]');
ylabel('Error [%]');

% Display identified parameters
fprintf('\nIdentified PI Controller Parameters:\n');
fprintf('Kp = %.4f\n', Kp);
fprintf('Ki = %.4f\n', Ki);

% Calculate fit quality
R2 = 1 - sum((duty_cycle - duty_cycle_pred).^2)/sum((duty_cycle - mean(duty_cycle)).^2);
fprintf('R² = %.4f\n', R2);

% Transfer function representation
s = tf('s');
C = Kp + Ki/s;
fprintf('\nController Transfer Function:\n');
fprintf('C(s) = %.4f + %.4f/s\n', Kp, Ki);