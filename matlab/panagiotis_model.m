clc;
clear;

% Load Data
data = readtable("../Data_for_MATLAB.xlsx",Range="B1:Q20894");
T_heat_coil = data.Heat_Coil_Temperature;
T_room = data.Room_Temperature;

% Data Parameters
dt = 1; % seconds
Cp = 1.005; 
T_amb = 77; % Fahrenheit
T_2nd_chamber = 95;
N = length(T_room);

% Build the linear system for the least squares problem:
A = zeros(N-1,2);
b = zeros(N-1,1);
for i = 2:N
    A(i-1,1) = (T_room(i) - T_room(i-1)) / dt; % Coefficient for Cap
    A(i-1,2) = 2*T_room(i) - (T_amb + T_2nd_chamber); % Coefficient for UA
    b(i-1) = -Cp*(T_room(i) - T_heat_coil(i-1));
end

% Solve via convex optimization (using CVX):
cvx_begin quiet
    variables Cap UA
    minimize( norm( A*[Cap; UA] - b, 2 ) )
    subject to
        Cap >= 0
        UA >= 0
cvx_end

disp(['Optimized Cap: ', num2str(Cap)]);
disp(['Optimized UA: ', num2str(UA)]);

% Simulate the system using the optimized parameters
T_predicted = zeros(N,1);
T_predicted(1) = T_room(1);
for i = 2:N
    T_predicted(i) = (Cap*T_predicted(i-1)/dt + Cp*T_heat_coil(i-1) + UA*(T_amb+T_2nd_chamber)) / (Cap/dt + Cp + 2*UA);
end

% Plot the results:
x = 1:N;
figure;
plot(x, T_room, x, T_predicted)
xlabel('Timestep')
ylabel('Temperature (F)')
legend('Data Temperature', 'Predicted Temperature')

% Other parameters
dt = 1;          % seconds
Cp = 1.005;
T_amb = 77;      % Fahrenheit (used in constant offset, not part of dynamic transfer function)
T_2nd_chamber = 95;

% Compute intermediate variables:
% The recurrence was:
%    T(k) = (Cap/dt * T(k-1) + Cp * T_heat_coil(k-1) + UA*(T_amb+T_2nd_chamber)) / (Cap/dt + Cp + 2*UA)
% For transfer function analysis we focus on the dynamics from T_heat_coil to T.
%
% Define:
%   alpha = (Cap/dt) / (Cap/dt + Cp + 2*UA)
%   beta  = Cp / (Cap/dt + Cp + 2*UA)
%
% Ignoring the constant offset term, the dynamic model is:
%   T(k) = alpha*T(k-1) + beta*T_heat_coil(k-1)
% which in the z-domain (with zero initial conditions) gives:
%   G(z) = (beta*z^(-1))/(1 - alpha*z^(-1))

Denom = (Cap/dt + Cp + 2*UA);
alpha = (Cap/dt) / Denom;
beta  = Cp / Denom;

% Create the discrete transfer function
num = [0 beta];  % the zero indicates a one-sample delay (z^-1)
den = [1 -alpha];
Gz = tf(num, den, dt);

% Display the transfer function
disp('Discrete Transfer Function G(z):');

% Stability Analysis:
% For a discrete system, the pole is at z = alpha.
p = pole(Gz);
disp('Poles of G(z):');
disp(p);

if all(abs(p) < 1)
    disp('The system is stable (all poles are inside the unit circle).');
else
    disp('The system is unstable (some poles lie outside the unit circle).');
end

% Plot the pole-zero map for a visual confirmation
figure;
pzmap(Gz);
title('Pole-Zero Map of G(z)');
