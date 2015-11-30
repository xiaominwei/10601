loaddata

%% Configuration
addpath minFunc;
rfSize = 6;
numCentroids=2000; % size of centoirds
whitening=true;
numPatches = 400000;
CIFAR_DIM=[32 32 3];

tic;

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) 
      fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  % unified distribution
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  % for each image, get numPatches/size(trainX,1) patches
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  % each line represents a small sample
  patches(i,:) = patch(:)';
end

% normalize for contrast
% ???????????????????????????????????10?
% bsxfun(@rdivide,A,B)??A???????B??????????
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% whiten
if (whitening)
    % the covariance matrix for patches
    C = cov(patches);
    M = mean(patches);
    [V,D] = eig(C);
    P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
    % the mean of each dimension should be zero
    % ??patches????????????????????????
    patches = bsxfun(@minus, patches, M) * P;
end

% run K-means
% ?????patches????????????centroids?
centroids = trainKmeans(patches, numCentroids, 50);
show_centroids(centroids, rfSize); drawnow;

% extract training features
if (whitening)
    % M??????P??????CIFAR_DIM?patch????rfSize??patch???
    trainXC = extractFeatures(trainX, centroids, rfSize, CIFAR_DIM, M,P);
else
    trainXC = extractFeatures(trainX, centroids, rfSize, CIFAR_DIM);
end

% standardize data
% ????svm???????????????
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
% ???????????????1
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

% train classifier using SVM
C = 100;
theta = trainSVM(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));
fprintf('Training time %f seconds \n ',toc);

%%%%% TESTING %%%%%

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);
testX = double(f1.data);
testY = double(f1.labels) + 1;
clear f1;

tic;

% compute testing features and standardize
if (whitening)
  testXC = extractFeatures(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extractFeatures(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));
fprintf('Test time %f seconds \n ',toc);
