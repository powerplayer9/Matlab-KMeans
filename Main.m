clc;
clear;
close all;

% Read images
img1 = imread('peppers.bmp');  % Insert the path to your image here
img2 = imread('lena.bmp');

% Select image to run
img = img2;

% K values and iteration count
K_values = [3, 5, 7];
iter = 50;
Costs = cell(1, numel(K_values));

% Run KMeans for each K
for i = 1:numel(K_values)
    K = K_values(i);
    Costs{i} = KMeans(K, img, iter);
end

% Plot costs
figure();
hold on;
colors = lines(numel(K_values));
for i = 1:numel(K_values)
    plot(Costs{i}, 'Color', colors(i,:));
end
grid on;
xlabel('Iteration Count');
ylabel('Cost');
legend(arrayfun(@(k) sprintf('K=%d', k), K_values, 'UniformOutput', false));
hold off;
