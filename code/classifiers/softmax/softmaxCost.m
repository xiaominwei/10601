function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);
M = exp(theta*data);
M = bsxfun(@rdivide, M, sum(M));
cost = (-1/size(data,2)).*sum(sum(groundTruth.*log(M)));
cost = cost + (lambda/2).*sum(sum(theta.^2));

thetagrad = (-1/size(data,2)).*(((groundTruth-(M))*data') + (lambda*theta));

grad = [thetagrad(:)];
end

