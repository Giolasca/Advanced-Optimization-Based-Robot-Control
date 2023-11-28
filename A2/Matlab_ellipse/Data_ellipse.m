% This script loads the points obtained from the Python simulations and
% saved in .mat files and generates the ellipse that best fits the data

%% Initialization
% Clear and close
clc; clear; close all;

%% Load the data
% Load the data from the .mat file
load('4500p-2q-2v.mat');

% Extract data from .mat
q_viable = table2array(points{1}(:, 1));          % viable q-coordinates
v_viable = table2array(points{1}(:, 2));          % viable v-coordinates
q_no_viable = table2array(points{2}(:, 1));     % non-viable q-coordinates
v_no_viable = table2array(points{2}(:, 2));    % non-viable v-coordinates

% Find the parameters of the ellipse that best fit the data
fit_ellipse = fit_ellipse(q_viable, v_viable);

% Draw the ellipse
figure;
plot(q_viable, v_viable, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'red');  
hold on;
plot(q_no_viable, v_no_viable, 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'blue');
draw_ellipse(fit_ellipse.a, fit_ellipse.b, fit_ellipse.phi, fit_ellipse.X0_in, fit_ellipse.Y0_in, 'k');
xlabel('q[rad]');
ylabel('dq [rad/s]');
legend('viable','non viable');

eq = sprintf('Equation of the ellipse: $\\frac{((x - \\pi) \\cos(%.2f) + y \\sin(%.2f))^2}{%.2f^2} + \\frac{((x - \\pi) \\sin(%.2f) - y \\cos(%.2f))^2}{%.2f^2} = 1$', ...
        fit_ellipse.phi, fit_ellipse.phi, fit_ellipse.a, fit_ellipse.phi, fit_ellipse.phi, fit_ellipse.b);

fprintf('Equation of the ellipse: ((x - %.4f) * cos(%.4f) + (y - %.4f) * sin(%.4f))^2 / %.4f^2 + ((x - %.4f) * sin(%.4f) - (y - %.4f) * cos(%.4f))^2 / %.4f^2 = 1\n', ...
        fit_ellipse.X0_in, fit_ellipse.phi, fit_ellipse.Y0_in, fit_ellipse.phi, fit_ellipse.a, ...
        fit_ellipse.X0_in, fit_ellipse.phi, fit_ellipse.Y0_in, fit_ellipse.phi, fit_ellipse.b);

% Saving image
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 24 16]); 
saveas(gcf,'4500p-2q-2v.jpg')


function draw_ellipse(a, b, phi, x0, y0, color)
    % Function to draw an ellipse
    t = linspace(0, 2*pi, 100);
    x = a * cos(t);
    y = b * sin(t);

    % Rotate the ellipse
    ell_rot = [cos(phi), -sin(phi); sin(phi), cos(phi)] * [x; y];

    % Translate the ellipse
    x = ell_rot(1, :) + x0;
    y = ell_rot(2, :) + y0;

    % Draw the ellipse
    plot(x, -y, color, 'LineWidth', 3);
end