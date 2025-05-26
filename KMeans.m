function [Cost] = KMeans(K, img, iter)
% KMeans - K-means clustering for image segmentation
%   K    - Number of Clusters
%   img  - Input Image in RGB color space
%   iter - Maximum Number of Iterations

% Reshape image into N x 3 data matrix
data = reshape(img, [], 3);

% Get data sizes
numPixels = size(data, 1);

% Initialize cluster centers randomly
Centres = randi([0 255], K, 3);

% Initialize cluster assignments
ClusterNumberOld = zeros(numPixels, 1);

% Prepare for visualization
imseg = zeros(size(img, 1), size(img, 2));
Cost = zeros(iter, 1);

figure();

% Original Image
subplot(2,2,1);
imshow(img);
title('Original Image');

% 3D plot of clusters in color space
p1 = subplot(2,2,3);
scatter3(p1, data(:,1), data(:,2), data(:,3), 1, zeros(numPixels,3));
title('Clusters in color space');
grid on;
xlabel('R'); ylabel('G'); zlabel('B');

% Segmented Image
p2 = subplot(2,2,2);
imagesc(imseg);
title(sprintf('Segmented Image; K = %d ; Iteration = 1', K));

% Cost plot
p3 = subplot(2,2,4);
plot(p3, Cost);
grid on;
title(sprintf('Cost; K = %d ; Iteration = 1', K));
xlabel('Iteration'); ylabel('Cost');

% K-means main loop
for i = 1:iter
    fprintf('K Iteration --> %d\t', i);

    % Compute distances to cluster centers
    Dist = pdist2(double(data), Centres);

    % Assign each pixel to the nearest cluster
    [DistShortest, ClusterNumber] = min(Dist, [], 2);

    % Compute cost (sum of distances)
    Cost(i) = sum(DistShortest);
    fprintf('Cost --> %d\n', Cost(i));

    % Update cluster centers
    for j = 1:K
        idx = (ClusterNumber == j);
        if any(idx)
            Centres(j,:) = mean(data(idx, :), 1);
        end
    end

    % Update segmented image
    imseg(:) = reshape(ClusterNumber, size(imseg));

    % Visualization
    z = double(ClusterNumber) / K;
    Z = [1-z, z, z];
    subplot(2,2,3);
    scatter3(p1, data(:,1), data(:,2), data(:,3), 1, Z);
    title('Clusters in color space');
    xlabel('R'); ylabel('G'); zlabel('B');

    subplot(2,2,2);
    set(p2, 'CData', imseg);
    title(sprintf('Segmented Image; K = %d ; Iteration = %d', K, i));

    subplot(2,2,4);
    plot(p3, Cost(1:i));
    title(sprintf('Cost; K = %d ; Iteration = %d', K, i));
    grid on;
    xlabel('Iteration'); ylabel('Cost');

    pause(0.1);

    % Early stopping if cost change is small
    if i >= 2 && abs(Cost(i-1) - Cost(i)) < 10
        Cost = Cost(1:i); % Trim unused entries
        break;
    end
end

end
