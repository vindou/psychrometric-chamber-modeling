%% Load Data
data = readtable('Data_for_MATLAB.xlsx', 'Range', 'B1:Q20894');
T_heat_coil = data.Heat_Coil_Temperature;
T_room      = data.Room_Temperature;
dt = 1;                  % time step in seconds
Cp = 1.005;              % heat capacity coefficient
T_amb = 77;              % ambient temperature (F)
T_2nd_chamber = 95;      % chamber setpoint (F)
N = length(T_room);

%% -----------------------------
%% TWO-NODE MODEL OPTIMIZATION
%% -----------------------------
% Two-node model: we now consider two states:
%   x1 = T_room,  x2 = T_chamber.
%
% We propose the following Euler–integrated dynamics:
%
%   T_room(k+1) = T_room(k) + dt/Cap_room * [ Cp*(T_heat_coil(k)-T_room(k)) ...
%                      + K*(T_chamber(k)-T_room(k)) + UA_amb*(T_amb - T_room(k)) ]
%
%   T_chamber(k+1) = T_chamber(k) + dt/Cap_chamber * [ K*(T_room(k)-T_chamber(k)) ...
%                          + UA_chamber*(T_2nd_chamber - T_chamber(k)) ]
%
% The parameter vector is: p = [UA_amb, K, Cap_room, Cap_chamber, UA_chamber].
% We assume T_chamber(1) = T_2nd_chamber (starting at its setpoint).

twoNodeObj = @(p) simulateTwoNode(p, T_heat_coil, T_room, dt, Cp, T_amb, T_2nd_chamber);

p0_two = [0.05, 0.1, 1200, 500, 0.05]; % initial guesses for the five parameters
lb_two = [0, 0, 0, 0, 0];
ub_two = [10, 10, 1e4, 1e4, 10];
opts_two = optimoptions('lsqnonlin','Display','iter');

p_opt_two = lsqnonlin(twoNodeObj, p0_two, lb_two, ub_two, opts_two);
fprintf('Optimized Two-Node Parameters:\n');
fprintf('  UA_amb    = %.4f\n  K         = %.4f\n  Cap_room  = %.2f\n  Cap_chamber = %.2f\n  UA_chamber= %.4f\n', ...
        p_opt_two(1), p_opt_two(2), p_opt_two(3), p_opt_two(4), p_opt_two(5));

% Simulate the two-node model with optimized parameters
[T_room_sim, T_chamber_sim] = simulateTwoNodeSim(p_opt_two, T_heat_coil, T_room, dt, Cp, T_amb, T_2nd_chamber);

figure;
plot(1:N, T_room, 'b', 1:N, T_room_sim, 'r--');
legend('Measured T_{room}', 'Two-Node Model');
xlabel('Time Step'); ylabel('Temperature (F)');
title('Two-Node Model Fit');

% Assume these are your optimized two-node parameters from lsqnonlin:
UA_amb    = p_opt_two(1);   % Coupling between ambient and room
K         = p_opt_two(2);   % Conduction between room and chamber
Cap_room  = p_opt_two(3);   % Heat capacity for the room node
Cap_chamber = p_opt_two(4); % Heat capacity for the chamber node
UA_chamber = p_opt_two(5);  % Coupling between chamber setpoint and chamber
Cp = 1.005;                 % Heat capacity coefficient for the heat coil

% Linearize the dynamics about the operating point (assuming T_amb and T_2nd_chamber are constant)
% and define the deviation variables:
%
%   Cap_room * dT_room/dt = Cp*u - (Cp + K + UA_amb)*T_room + K*T_chamber
%   Cap_chamber * dT_chamber/dt = K*T_room - (K + UA_chamber)*T_chamber
%
% The state-space matrices then become:
A = [ -(Cp+K+UA_amb)/Cap_room,    K/Cap_room;
      K/Cap_chamber,             -(K+UA_chamber)/Cap_chamber ];
B = [ Cp/Cap_room; 0 ];
C = [ 1, 0 ];  % Output is T_room deviation
D = 0;

% Create the state-space system and convert it to a transfer function:
sys_ss = ss(A, B, C, D);
G_s = tf(sys_ss);

disp('Linearized transfer function from heat coil input to room temperature:');
G_s


%% -----------------------------
%% LOCAL FUNCTIONS

%% Two-node simulation that returns error (for optimization)
function err = simulateTwoNode(p, T_heat_coil, T_room, dt, Cp, T_amb, T_2nd_chamber)
    % Unpack parameters
    UA_amb    = p(1);  % coupling between ambient and room
    K         = p(2);  % conduction between room and chamber
    Cap_room  = p(3);
    Cap_chamber = p(4);
    UA_chamber = p(5); % coupling between chamber setpoint and chamber
    N = length(T_room);
    T_room_sim = zeros(N,1);
    T_chamber_sim = zeros(N,1);
    T_room_sim(1) = T_room(1);    % initial measured room temperature
    T_chamber_sim(1) = T_2nd_chamber;  % assume chamber starts at its setpoint
    for k = 1:N-1
        T_room_sim(k+1) = T_room_sim(k) + dt/Cap_room * (Cp*(T_heat_coil(k) - T_room_sim(k)) ...
                              + K*(T_chamber_sim(k) - T_room_sim(k)) + UA_amb*(T_amb - T_room_sim(k)));
        T_chamber_sim(k+1) = T_chamber_sim(k) + dt/Cap_chamber * (K*(T_room_sim(k) - T_chamber_sim(k)) ...
                              + UA_chamber*(T_2nd_chamber - T_chamber_sim(k)));
    end
    err = T_room_sim - T_room;
end

%% Two-node simulation that returns the simulated room and chamber temperatures
function [T_room_sim, T_chamber_sim] = simulateTwoNodeSim(p, T_heat_coil, T_room, dt, Cp, T_amb, T_2nd_chamber)
    UA_amb    = p(1);
    K         = p(2);
    Cap_room  = p(3);
    Cap_chamber = p(4);
    UA_chamber = p(5);
    N = length(T_room);
    T_room_sim = zeros(N,1);
    T_chamber_sim = zeros(N,1);
    T_room_sim(1) = T_room(1);
    T_chamber_sim(1) = T_2nd_chamber;
    for k = 1:N-1
        T_room_sim(k+1) = T_room_sim(k) + dt/Cap_room * (Cp*(T_heat_coil(k) - T_room_sim(k)) ...
                              + K*(T_chamber_sim(k) - T_room_sim(k)) + UA_amb*(T_amb - T_room_sim(k)));
        T_chamber_sim(k+1) = T_chamber_sim(k) + dt/Cap_chamber * (K*(T_room_sim(k) - T_chamber_sim(k)) ...
                              + UA_chamber*(T_2nd_chamber - T_chamber_sim(k)));
    end
end
