function [W, Z, output] = sccadeflate(trainX,Kb,k,a,b,c)

%
% This is a wrapper function that runs the SCCA2 function while deflating
% and finding new projection directions
%
% Input:
%   trainX - first view train data
%   Kb     - second view train data
%   k      - vector of indices for deflations
%   a,b    - output and debug variables for scca
%
% Output:
%   W      - Projection for primal
%   Z      - Projection for dual
%   output - a struct with various values
%
% Written by David R. Hardoon
% UCL D.Hardoon@cs.ucl.ac.uk

tX = trainX;
KK = Kb;
co = 1;

for i=1:length(k)
    fprintf('.');
    [output.w,output.e,t1,t2,t3,t4,output.cor,output.res] = scca(tX,KK,k(i),a,b,c);
    
    wa(:,co) = output.w;
    e(:,co) = output.e;
    resval(co) = output.res;
    corval(co) = output.cor;
    co = co + 1;
    
    % Dual Deflation 
    projk(:,i) = KK*e(:,i);
    tau(:, i) = KK*projk(:,i);

    P = eye(length(KK)) - (tau(:,i)*tau(:,i)')/(tau(:,i)'*tau(:,i));    
    KK = P'*KK*P;

    % Primal Deflation
    proj(:,i) = tX*(tX'*wa(:,i));
    t(:,i) = tX'*proj(:,i);
    tX = tX - tX*(t(:,i)*t(:,i)')/(t(:,i)'*t(:,i));
end
disp(' ');

% Primal projection
P = trainX*t*inv(t'*t);
W = (proj*inv(P'*proj));

% can't think of a fancy way to normalise the vectors
for i=1:size(W,2)
    WW(:,i) = W(:,i)/norm(trainX'*W(:,i));
end
W = WW;

% Dual Projection
Z = projk*inv(inv(tau'*tau)*tau'*Kb*projk);

for i=1:size(Z,2)
    ZZ(:,i) = Z(:,i)/norm(Kb*Z(:,i));
end
Z = ZZ;
    
output = [];
output.primal.w = wa;
output.dual.e = e;
output.primal.P = P;
output.dual.tau = tau;
output.primal.tau = t;
output.cor = corval;
output.res = resval;
output.W = W;
output.Z = Z;