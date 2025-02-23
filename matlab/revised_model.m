clc;
clear;

function err = chamberResidual(params, T_room_data, T_coil_data, ...
                               T_amb, T_2nd, dt, Cp)
    % Unpack parameters
    Cap    = params(1);
    UAamb  = params(2);
    UA2ch  = params(3);

    N = length(T_room_data);
    T_model = zeros(N,1);
    T_model(1) = T_room_data(1);  % match initial condition

    for k = 2:N
        dTdt = ( Cp*(T_coil_data(k-1) - T_model(k-1)) ...
               + UAamb*(T_amb - T_model(k-1)) ...
               + UA2ch*(T_2nd - T_model(k-1)) ) / Cap;
        T_model(k) = T_model(k-1) + dTdt*dt;
    end

    % Residual = difference between measured and modeled
    err = T_room_data - T_model;
end


% 1) Load data
data = readtable("Data_for_MATLAB.xlsx",Range="B1:Q20894");
T_room_data     = data.Room_Temperature;        % measured room temperature
T_coil_data     = data.Heat_Coil_Temperature;   % measured coil temperature
N = length(T_room_data);

% 2) Define known constants
dt   = 1;       % [s]
Cp   = 1.005;   % air heat capacity
T_amb = 77;     % ambient T (F)
T_2nd = 95;     % 2nd chamber T (F)

% 3) Create a function that, given candidate params [Cap, UAamb, UA2ch],
%    simulates the model and returns the residual (difference) at each step.

modelResidual = @(params) chamberResidual(params, T_room_data, T_coil_data, ...
                                          T_amb, T_2nd, dt, Cp);

% 4) Initial guesses and bounds for [Cap, UAamb, UA2ch]
params0 = [1000;  0.05;  0.05];  % example guesses
lb      = [  0;  0;     0   ];   % enforce non-negativity
ub      = [Inf; Inf;   Inf ];

options = optimoptions('lsqnonlin','Display','iter');
params_opt = lsqnonlin(modelResidual, params0, lb, ub, options);

Cap_opt    = params_opt(1);
UAamb_opt  = params_opt(2);
UA2ch_opt  = params_opt(3);

disp('--- Optimized Parameters ---');
disp(['Cap        = ', num2str(Cap_opt)]);
disp(['UA_amb     = ', num2str(UAamb_opt)]);
disp(['UA_2ch     = ', num2str(UA2ch_opt)]);

% 7) Now simulate with these optimized parameters and plot
T_pred = zeros(N,1);
T_pred(1) = T_room_data(1);  % initial condition
for k = 2:N
    % Using the revised model:
    dTdt = ( Cp*(T_coil_data(k-1) - T_pred(k-1)) ...
           + UAamb_opt*(T_amb - T_pred(k-1)) ...
           + UA2ch_opt*(T_2nd - T_pred(k-1)) ) / Cap_opt;
    T_pred(k) = T_pred(k-1) + dTdt*dt;
end

figure; hold on;
plot(T_room_data,'b','DisplayName','Data Temperature');
plot(T_pred,'r','DisplayName','Predicted Temperature');
xlabel('Timestep');
ylabel('Temperature (F)');
legend('Location','best');
title('Revised Model Fit');


% Other known constants:
dt = 1;       % sampling time (s or min)
Cp = 1.005;   % air heat capacity

%--------------------------------------------------------------------------
% 1) Derive the single-pole discrete model ignoring offsets:
%
%   T(k+1) = a * T(k) + b * T_coil(k) convert into continuous time
%
% where
%   a = 1 - (dt/Cap_opt)*(Cp + UAamb_opt + UA2ch_opt)
%   b = (dt * Cp) / Cap_opt
%
% The stability condition is |a| < 1.

a = 1 - (dt/Cap_opt)*(Cp + UAamb_opt + UA2ch_opt);
b = (dt*Cp)/Cap_opt;

fprintf('Computed pole a = %.4f\n', a);

if abs(a) < 1
    disp('=> The system is stable (|a| < 1).');
else
    disp('=> The system is UNSTABLE (|a| >= 1).');
end

%--------------------------------------------------------------------------
% 2) Create a discrete transfer function from T_coil to T
%    G(z) = (b * z^-1) / (1 - a * z^-1).

num = [0 b];  % "0" enforces the one-step delay (z^-1)
den = [1 -a];
Gz  = tf(num, den);

disp('Discrete transfer function from T_coil to T_room:');
Gz

%--------------------------------------------------------------------------
% 3) Pole-zero map in MATLAB
figure;
pzmap(Gz);
title('Pole-Zero Map of Revised Single-State Model');
grid on;

p = pole(Gz);
fprintf('Pole(s) of G(z): %g\n', p);

