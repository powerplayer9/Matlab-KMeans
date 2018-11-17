clc;
clear all;
close all;


img1 = imread('peppers.bmp');
img2 = imread('lena.bmp');
% K =3;
iter = 50;

%% KM
%[Cost] = KMeans(K,img,iter)
% Selecting Image to Run
img = img2;

%% K = 3
K = 3;
[Cost3K] = KMeans(K,img,iter);

%% K = 5
K = 5;
[Cost5K] = KMeans(K,img,iter);

%% K = 7
K = 7;
[Cost7K] = KMeans(K,img,iter);

%% Cost plot
figure();
plot(Cost3K);
hold on; plot(Cost5K);
hold on; plot(Cost7K);
grid on;
xlabel('Iteration Count'); ylabel('Cost');
legend('K=3','K=5','K=7');


