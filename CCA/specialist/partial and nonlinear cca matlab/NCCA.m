function [Xproj, Yproj, cor, XunpairedProj,YunpairedProj, OSEoutputs] = ...
    NCCA(X, Y, d, Xunpaired, Yunpaired, params)
% NCCA: Nonparametric canonical correlation analysis
%
% Input:
% X,Y - paired training examples (rows) of features (columns).
% d - dimension of the output transformed features.
% X_unpaired, Y_unpaired [optional] - additional unpaired examples. The
%       numbers of unpaired X and Y examples does not need to be identical.
% params [optional] - structure with algorithm paramters:
%       - hx,hy - bandwidth parameters for the KDEs of views 1,2. Default
%       is 0.5.
%       - nnx,nny - number of nearest neighbors for the KDEs of views 1,2.
%       Default is 20.
%       - randSVDiters - number of iterations for random SVD algorithm.
%       Set higher for better accuracy, default is 20.
%       - randSVDblock - block size for random SVD algorithm. Set higher
%       for better accuracy, default is d+10;
%       - doublyStIters - number of iterations for doubly stichastic
%       normalization. Set higher for better accuracy, default is 15.
%
%
% Output:
% Xproj,Yproj - d-dimensional projections of the training data.
% cor - correlations between the d pairs of projections.
% XunpairedProj,YunpairedProj [optional] - d-dimensional projections of
%       the unpaired data.
% OSEoutputs [optional] - outputs needed for out-of-sample extension
%
%
% This code is based on the paper:
%       Tomer Michaeli, Weiran Wang and Karen Livescu,
%       "Nonparametric Canonical Correlation Analysis",
%       International Conference on Machine Learning (ICML 2016)
%
% Written by Tomer Michaeli (Technion - IIT) and Weiran Wang (TTIC).
%
% This Matlab code is distributed only for academic research purposes.
% For other purposes, please contact Tomer Michaeli
% mail: tomer.m@ee.technion.ac.il
%
% Version 3.0, June 2017.


%% Check inputs and set default values to the unspecified parameters
narginchk(3,6);

if ~exist('Xunpaired','var')
    Xunpaired = [];
end
if ~exist('Yunpaired','var')
    Yunpaired = [];
end
if ~exist('params','var')
    params = [];
end
if ~isfield(params,'hx')
    hx = 0.5;
else
    hx = params.hx;
end
if ~isfield(params,'hy')
    hy = 0.5;
else
    hy = params.hy;
end
if ~isfield(params,'nnx')
    nnx = 20;
else
    nnx = params.nnx;
end
if ~isfield(params,'nny')
    nny = 20;
else
    nny = params.nny;
end
if ~isfield(params,'doublyStIters')
    doublyStIters = 15;
else
    doublyStIters = params.doublyStIters;
end
if ~isfield(params,'randSVDiters')
    randSVDiters = 20;
else
    randSVDiters = params.randSVDiters;
end
if ~isfield(params,'randSVDblock')
    randSVDblock = d+10;
else
    randSVDblock = params.randSVDblock;
end

N = size(X,1); % Number of paired training points
NXun = size(Xunpaired,1); % Number of unpaired X training points
NYun = size(Yunpaired,1); % Number of unpaired Y training points


%% Normalize data
meanX = mean(X); meanY = mean(Y);
X = bsxfun(@minus, X, meanX);
Y = bsxfun(@minus, Y, meanY);

meanSqX = sqrt(mean(sum(X.^2,2))); meanSqY = sqrt(mean(sum(Y.^2,2)));
X = bsxfun(@times, X, 1./meanSqX);
Y = bsxfun(@times, Y, 1./meanSqY);

if NXun>0
    Xunpaired = bsxfun(@minus, Xunpaired, meanX);
    Xunpaired = bsxfun(@times, Xunpaired, 1./meanSqX);
end
if NYun>0
    Yunpaired = bsxfun(@minus, Yunpaired, meanY);
    Yunpaired = bsxfun(@times, Yunpaired, 1./meanSqY);
end

X = [X ; Xunpaired];
Y = [Y ; Yunpaired];
clear Xunpaired Yunpaired


%% Compute NNs
fprintf('Computing nearest neighbors ...'); tic;

[idxs_X,dists_X] = knnsearch(X(1:N,:), X, 'K', nnx);
dists_X = dists_X.^2'; idxs_X = idxs_X';
[idxs_Y,dists_Y] = knnsearch(Y(1:N,:), Y, 'K', nny);
dists_Y = dists_Y.^2'; idxs_Y = idxs_Y';

toc; clear X Y


%% Compute weight matrices for the two views
colInd = ones(nnx, 1) * (1:(N+NXun));
Dx = sparse(double(idxs_X(:)), colInd(:), exp(-0.5*dists_X(:)/hx^2), N, N+NXun)';
colInd = ones(nny, 1) * (1:(N+NYun));
Dy = sparse(double(idxs_Y(:)), colInd(:), exp(-0.5*dists_Y(:)/hy^2), N, N+NYun);

clear colInd idxs_X idxs_Y dists_X dists_Y


%% Normalize the weight matrices
Dx = spdiags(1./sum(Dx,2), 0, N+NXun, N+NXun) * Dx;
Dy = Dy * spdiags(1./sum(Dy,1)', 0, N+NYun, N+NYun);


%% Doubly stochastic normalization
onesVecX = ones(N+NXun,1);
onesVecY = ones(N+NYun,1);

fprintf('Normalizing S to be doubly stochastic ...'); tic;
for its=1:doublyStIters
    Dy = Dy * spdiags(1./((onesVecX'*Dx)*Dy)', 0, N+NYun, N+NYun) * ((N+NXun)/N);
    Dx = ((N+NXun)/N) * spdiags(1./(Dx*(Dy*onesVecY)), 0, N+NXun, N+NXun) * Dx;
end

toc; clear onesVecX onesVecY


%% Compute projections using the SVD of the kernel
fprintf('Computing SVD ...');

tic; [U,D,V] = randpca_AB(Dx, Dy, d+1, randSVDiters, randSVDblock); toc;
Xproj = U(1:N,2:end) * sqrt(N+NXun);
Yproj = V(1:N,2:end) * sqrt(N+NYun);

if NXun>0
    XunpairedProj = U(N+1:end,2:end) * sqrt(N+NXun);
else
    XunpairedProj = [];
end
if NYun>0
    YunpairedProj = V(N+1:end,2:end) * sqrt(N+NYun);
else
    YunpairedProj = [];
end
cor = diag(D(2:end,2:end));


%% Assigning outputs needed for out-of-sample extension
OSEoutputs.Dx = Dx;
OSEoutputs.Dy = Dy;
OSEoutputs.hx = hx;
OSEoutputs.hy = hy;
OSEoutputs.nnx = nnx;
OSEoutputs.nny = nny;
OSEoutputs.meanX = meanX;
OSEoutputs.meanY = meanY;
OSEoutputs.meanSqX = meanSqX;
OSEoutputs.meanSqY = meanSqY;
OSEoutputs.Ntr = N;
OSEoutputs.d = d;

end
