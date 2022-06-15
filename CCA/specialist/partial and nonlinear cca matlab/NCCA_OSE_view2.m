function [YnewProj] = NCCA_OSE_view2(Ynew,Ytr,Xproj,cor,OSEoutputs)
% Nystrom out-of-sample extension for view 2 obtained with NCCA
%
% Input:
% Ynew - new examples (rows) of features (columns).
% Ytr - original training examples of view 2 which were used to obtain 
%       the projections with NCCA.
% Xproj - the projections of the view 1 examples which were obtained with
%       NCCA.
% cor - the correlations between the projections which were obtained with
%       NCCA.
% OSEoutputs - output structure from the NCCA function, needed for
%       out-of-sample extension.
%
% Output:
% YnewProj - d-dimensional projections of the new examples.
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


N = size(Ynew,1);

%% Normalize data
Ynew = bsxfun(@minus, Ynew, OSEoutputs.meanY);
Ynew = bsxfun(@times, Ynew, 1./OSEoutputs.meanSqY);
Ytr = bsxfun(@minus, Ytr, OSEoutputs.meanY);
Ytr = bsxfun(@times, Ytr, 1./OSEoutputs.meanSqY);

%% Compute NNs for new samples
[idxs_Y,dists_Y] = knnsearch(Ytr, Ynew, 'K', OSEoutputs.nny);
dists_Y = dists_Y.^2'; idxs_Y = idxs_Y';
clear Ynew Ytr

%% Computing new entries in weight matrix & normalizing
colInd = ones(OSEoutputs.nny,1) * (1:N);
Dy = sparse(double(idxs_Y(:)), colInd(:), exp(-0.5*dists_Y(:)/OSEoutputs.hy^2), OSEoutputs.Ntr, N);
Dy = Dy * spdiags(1./sum(Dy,1)', 0, N, N);
clear colInd dists_Y idxs_Y

%% Column normalization
onesVec = ones(OSEoutputs.Ntr,1);
Dy = Dy * spdiags(1./((onesVec'*OSEoutputs.Dx)*Dy)', 0, N, N);
clear onesVec

%% Computing new projections with out-of-sample extension
YnewProj = (OSEoutputs.Dx * Dy)' * Xproj;
YnewProj = bsxfun(@times, YnewProj, 1./cor); 

end

