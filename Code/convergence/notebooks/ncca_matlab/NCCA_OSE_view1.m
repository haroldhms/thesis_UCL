function [XnewProj] = NCCA_OSE_view1(Xnew,Xtr,Yproj,cor,OSEoutputs)
% Nystrom out-of-sample extension for view 1 obtained with NCCA
%
% Input:
% Xnew - new examples (rows) of features (columns).
% Xtr - original training examples of view 1 which were used to obtain 
%       the projections with NCCA.
% Yproj - the projections of the view 2 examples which were obtained with
%       NCCA.
% cor - the correlations between the projections which were obtained with
%       NCCA.
% OSEoutputs - output structure from the NCCA function, needed for
%       out-of-sample extension.
%
% Output:
% XnewProj - d-dimensional projections of the new examples.
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
% Version 2.0, January 2017.


N = size(Xnew,1);

%% Normalize data
Xnew = bsxfun(@minus, Xnew, OSEoutputs.meanX);
Xnew = bsxfun(@times, Xnew, 1./OSEoutputs.meanSqX);
Xtr = bsxfun(@minus, Xtr, OSEoutputs.meanX);
Xtr = bsxfun(@times, Xtr, 1./OSEoutputs.meanSqX);

%% Compute NNs for new samples
[idxs_X,dists_X] = knnsearch(Xtr, Xnew, 'K', OSEoutputs.nnx);
dists_X = dists_X.^2'; idxs_X = idxs_X';
clear Xnew Xtr

%% Computing new entries in weight matrix & normalizing
colInd = ones(OSEoutputs.nnx,1) * (1:N);
Dx = sparse(double(idxs_X(:)), colInd(:), exp(-0.5*dists_X(:)/OSEoutputs.hx^2), OSEoutputs.Ntr, N)';
Dx = spdiags(1./sum(Dx,2), 0, N, N) * Dx;
clear colInd dists_X idxs_X

%% Row normalization
onesVec = ones(OSEoutputs.Ntr,1);
Dx = spdiags(1./(Dx*(OSEoutputs.Dy*onesVec)), 0, N, N) * Dx;
clear onesVec

%% Computing new projections with out-of-sample extension
XnewProj = Dx * OSEoutputs.Dy * Yproj;
XnewProj = bsxfun(@times, XnewProj, 1./cor); 

end

