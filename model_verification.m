clear; clc; close all;

%% 1. Load experimental data
% Suppose your Excel file has three columns: time (sec), input, output
data = xlsread('Data_for_MATLAB.xlsx');
time_exp = data(:,1);   % Time vector
u_exp    = data(:,9);   % Measured input (e.g. reference signal R)
y_exp    = data(:,3);   % Measured output (the temperature response, etc.)

%% 2. Define transfer functions in MATLAB
% Create the Laplace variable s
s = tf('s');

% Controller: parallel PI
C = 10 + 0.0015/s;

% Plant
P = 1.0012/(15.86*s + 1.322);

% Sensor dynamics (assumed L(s) = 1 in your notes)
L = 1;

% For standard negative feedback with unity feedback path L(s) = 1,
% the closed-loop from R(s) to Y(s) is:
%         Gcl(s) = [C(s)*P(s)] / [1 + C(s)*P(s)*L(s)]
% Here, L = 1, so:
G_cl = (C*P)/(1 + C*P*L);

%% 3. Simulate the model response
% Use the same time vector and input signal as in the experiment
% lsim() applies u_exp to G_cl and returns the simulated output
y_model = lsim(G_cl, u_exp, time_exp);

%% 4. Plot experimental vs model output
figure; 
plot(time_exp, y_exp, 'b', 'LineWidth', 1.5); hold on;
plot(time_exp, y_model, 'r--', 'LineWidth', 1.5);
grid on; xlabel('Time (s)');
ylabel('Output');
legend('Measured Output','Model Output','Location','Best');
title('Comparison of Experimental Data vs. Closed-Loop Model');

%% 5. Compute an error metric (optional)
% e.g. root mean squared error between measured and modeled outputs
rmse = sqrt(mean((y_exp - y_model).^2));
disp(['RMSE = ', num2str(rmse)]);
