% Define the coefficients of the polynomial
% 15.86s³ + 17.182s² + 11.334s + 0.001502 = 0
coeff = [15.86 17.182 11.334 0.001502];

% Solve for the roots (poles)
poles = roots(coeff);

% Display results
fprintf('Closed-loop poles:\n');
for i = 1:length(poles)
    fprintf('s%d = %.6f + %.6fi\n', i, real(poles(i)), imag(poles(i)));
end

% Check stability
stable = all(real(poles) < 0);
if stable
    fprintf('\nThe system is stable (all poles have negative real parts)\n');
else
    fprintf('\nThe system is unstable (at least one pole has positive real part)\n');
end

% Plot poles
figure;
plot(real(poles), imag(poles), 'x', 'MarkerSize', 10);
grid on;
hold on;
plot([-1 1]*max(abs(real(poles))), [0 0], 'k--'); % Real axis
plot([0 0], [-1 1]*max(abs(imag(poles))), 'k--'); % Imaginary axis
title('Pole Locations');
xlabel('Real Axis');
ylabel('Imaginary Axis');
axis equal;
