function [K] = gaussian_kernel(X)

% this is a function to obtain the gaussian kernel matrix
% given an input matrix

nsq=sum(X.^2,2);
K=bsxfun(@minus,nsq,(2*X)*X.');
K=bsxfun(@plus,nsq.',K);
K=exp(-K);