% Read the .png image
img = imread('4500p_1g_random.png'); % Replace with the path to your image file

% Extract color channels (RGB)
redChannel = img(:, :, 1);
greenChannel = img(:, :, 2);
blueChannel = img(:, :, 3);

% Find red-colored pixels
redPixels = redChannel > 200 & greenChannel < 100 & blueChannel < 100;

% Find blue-colored pixels
bluePixels = redChannel < 100 & greenChannel < 100 & blueChannel > 200;

% Get coordinates of red-colored pixels
[rowRed, colRed] = find(redPixels);

% Get coordinates of blue-colored pixels
[rowBlue, colBlue] = find(bluePixels);

% Display the original image and mark red and blue points
figure;
imshow(img);
hold on;
plot(colRed, rowRed, 'ro', 'MarkerSize', 1.8); % Mark red points on the figure
plot(colBlue, rowBlue, 'bo', 'MarkerSize', 1.8); % Mark blue points on the figure

% Now, the variables 'rowRed', 'colRed', 'rowBlue', 'colBlue' 
% contain the coordinates of red and blue points


