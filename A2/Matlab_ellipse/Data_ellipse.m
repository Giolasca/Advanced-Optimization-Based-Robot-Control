% This script loads the points obtained from the Python simulations and
% saved in .mat files and generates the ellipse that best fits the data

%% Initialization
% Clear and close
clc; clear; close all;

%% Load the data
% File's names
file_name = {'4500p-5g', '4500p-2q-2v'}; 

% User select a file
Index_file = listdlg('PromptString', {'Select a file to extract data from:', ''}, ...
                    'ListString', file_name, 'SelectionMode', 'single');

if isempty(Index_file)
    disp('No selection made. The program terminates.');
else
    switch Index_file
    case {1} 
        load('4500p-5g.mat');
    case {2}
        load('4500p-2q-2v.mat');
    otherwise
        error('Wrong selection battery.');
    end
end

% Extract data from .mat
q_viable = table2array(points{1}(:, 1));          % viable q-coordinates
v_viable = table2array(points{1}(:, 2));          % viable v-coordinates
q_no_viable = table2array(points{2}(:, 1));     % non-viable q-coordinates
v_no_viable = table2array(points{2}(:, 2));    % non-viable v-coordinates

%% Fit-Ellipse
% Find the parameters of the ellipse that best fit the data
fit_ellipse = fit_ellipse(q_viable, v_viable);

% Draw the ellipse
figure;
plot(q_viable, v_viable, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'red');  
hold on;
plot(q_no_viable, v_no_viable, 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'blue');
draw_ellipse(fit_ellipse.a, fit_ellipse.b, fit_ellipse.phi, fit_ellipse.X0_in, fit_ellipse.Y0_in, 'k');
xlabel('q[rad]');
ylabel('dq [rad/s]');
legend('viable','non viable', 'ellipse');

xlim([min(q_no_viable(:,1))-0.15, max(q_no_viable(:,1))+0.15]);
ylim([min(v_no_viable(:,1))-1, max(v_no_viable(:,1))+1]);

% Calculate the area of the ellipse
ellipse_area = pi * fit_ellipse.a * fit_ellipse.b;

%% Plot and ellipse information
% Display information about the ellipse
fprintf('Major axis length: %.4f\n', fit_ellipse.a);
fprintf('Minor axis length: %.4f\n', fit_ellipse.b);
fprintf('Orientation angle: %.4f radians\n', fit_ellipse.phi);
fprintf('Area of the ellipse: %.4f\n', ellipse_area);

% Display the equation of ellipse
fprintf('Equation of the ellipse: ((x - π) * cos(%.4f) + (y) * sin(%.4f))^2 / %.4f^2 + ((x - π) * sin(%.4f) - (y) * cos(%.4f))^2 / %.4f^2 = 1\n', ...
        fit_ellipse.phi, fit_ellipse.phi, fit_ellipse.a, fit_ellipse.phi, fit_ellipse.phi, fit_ellipse.b);

% Saving image
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 24 16]); 
saveas(gcf,file_name{Index_file})


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