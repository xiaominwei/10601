

clear all;

CIFAR_DIR='data/train/';

assert(strcmp(CIFAR_DIR, 'cifar-10-batches-mat/'), ...%strcmp????1
       ['You need to modify the address so that CIFAR_DIR points to ' ...
        'your cifar-10-batches-mat directory.  You can download this ' ...
        'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);

%% Configuration


%% Load CIFAR training data

fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);%50000*3072
trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!,??1?10
clear f1 f2 f3 f4 f5;
fprintf('Loading Done\n');

%% Data to Image

% Red = trainX(3,1:1024);
% Green = trainX(3,1025:2048);
% Blue = trainX(3,2049:3072);
% 
% image(:,:,1) = reshape(Red, 32, 32);
% image(:,:,2) = reshape(Green, 32, 32);
% image(:,:,3) = reshape(Blue, 32, 32);
% 
% imshow(image/256);
