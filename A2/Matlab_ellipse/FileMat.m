% Clear and close
clc; clear; close all;

% Load the two .mat files
load('viable.mat');
load('no_viable.mat');

% Assign column names (modify these names according to your needs)
column_names = {'q_viable', 'dq_viable'};
column_names_q = {'q_no_viable', 'dq_no_viable'};

% Create a 1x2 cell containing the columns from the two files
points = cell(1, 2);

% Assign column names for the first file
table1 = table(viable_states(:, 1), viable_states(:, 2), 'VariableNames', column_names);

% Assign column names for the second file
table2 = table(no_viable_states(:, 1), no_viable_states(:, 2), 'VariableNames', column_names_q);

% Insert the tables into the cell
points{1} = table1;
points{2} = table2;

% Save the cell into a .mat file
save('4500p-5q.mat', 'points');