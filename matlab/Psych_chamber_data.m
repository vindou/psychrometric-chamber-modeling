clc
clear all

data = readtable("Data_for_MATLAB.xlsx",Range="B1:Q20894");
T_heat_coil = data.Heat_Coil_Temperature;
T_room = data.Room_Temperature;

%Data
dt = 1; %seconds
Cp = 1.005; 
T_amb = 77; %Fahrenheit
T_2nd_chamber = 95;

for i = 1:20893
    Cp_DT(i) = Cp*(T_heat_coil(i)-T_room(i)); %Cp*(T_supply-T_room)
    DT_amb_room(i) = T_amb-T_room(i);
    DT_room_chamber(i) = T_2nd_chamber-T_room(i);
end

%The 2 values below are guesses, but they follow a logic. Using the chamber
%energy equation, at the beginning the temp is relatively stable, which
%means we can get an estimation for the UA term since the ΔΤ/dt will be 0.
%Using this value, we go to the more transient region where we can make an
%approximation for the Cap value. (Of course, we take an average value for
%both UA and Cap for the "region" where we approximate them since they
%have variability). After this initial guess we just play around to find
%values that fit well with the data

UA = 0.05; %Guess, its divided by m_air
Cap = 1200; %Guess, its divided by m_air

T_predicted(1) = T_room(1);

for i=2:20893
    T_predicted(i) = (Cap*T_predicted(i-1)/dt + Cp*T_heat_coil(i-1)+UA*(T_amb+T_2nd_chamber))/(Cap/dt+Cp+2*UA);
end


x = 1:20893;
plot(x,T_room,x,T_predicted)
xlabel('Timestep')
ylabel('Temperature (F)')
legend('Data Temperature', 'Predicted Temperature')

%The fit is not gonna be perfect, but we want to have something that at
%least captures the overall trends of the curve to start working

