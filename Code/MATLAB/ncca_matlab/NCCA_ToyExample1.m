close all
clear

%% Parameters
N=1000; % Overal number of examples (paired+unpaired)
N_paired = 800; % Number of paired examples
MaxAngle = 4*pi;
MinRadius = 0.3;
MaxRadius = 8;
params.nnx = 10; % Number of nearest neighbors for view 1 KDE
params.nny = 10; % Number of nearest neighbors for view 2 KDE
d = 1; % dimension of the projection
rng(8409);  % For getting reproducible simulations

%% Generate data for views 1,2
t = linspace(0,MaxAngle,N);
r = linspace(MinRadius,MaxRadius,N) + 2*rand(1,N);
X = [r.*cos(t+0*randn(size(t))*0.05); r.*sin(t+0*randn(size(t))*0.05)] + 0*randn(2,length(t));
Y = [t+0*randn(size(t))*1; 2*randn(size(t))]+0*[zeros(1,N); randn(1,N)];
X = X'; Y = Y';

% Training data
PairedIndices = randperm(N);
PairedIndices = PairedIndices(1:N_paired);
Xpointers = zeros(1,N);
Xpointers(PairedIndices) = PairedIndices; % If Xpointers(i)=j then the i'th X point is matched to the j'th Y point

% Test (or validation) data
UnpairedIndices = setdiff(1:N,PairedIndices);

%% Plot data
figure(1)
subplot(2,2,1), cla, scatter(X(:,1), X(:,2), 20, t(1:N)', 'filled'), title('View 1')
axis equal
axis tight
hold on, scatter(X(PairedIndices,1),X(PairedIndices,2), 40, 0*t(PairedIndices)')
legend('Paired', 'Unpaired')

figure(1)
subplot(2,2,2), cla, scatter(Y(:,1),Y(:,2),20, t(1:N)', 'filled'), title('View 2')
axis equal
axis tight
hold on, scatter(Y(Xpointers(PairedIndices),1), Y(Xpointers(PairedIndices),2), 40, 0*t(PairedIndices)')
legend('Paired', 'Unpaired')

%% Run nonparametric CCA with only paired examples
fprintf('Running NCCA on paired examples:\n');
[XprojPaired, YprojPaired, cor, ~, ~, OSEoutputs] = ...
  NCCA(X(PairedIndices,:), Y(Xpointers(PairedIndices),:), d, [], [], params);

%% Run Nystrom out-of-sample extension on unpaired examples
fprintf('\nRunning Nystrom out-of-sample extension...'); tic;
[XprojNystrom] = NCCA_OSE_view1(X(UnpairedIndices,:),X(PairedIndices,:),YprojPaired,cor,OSEoutputs);
[YprojNystrom] = NCCA_OSE_view2(Y(setdiff(1:N,Xpointers(PairedIndices)),:),Y(Xpointers(PairedIndices),:),XprojPaired,cor,OSEoutputs);
toc;

%% Run nonparametric CCA on all (paired + unpaired) examples
fprintf('\nRunning NCCA on all examples:\n');
[XprojPaired2, YprojPaired2, cor, XprojUnpaired2, YprojUnpaired2, OSEoutputs] = ...
  NCCA(X(PairedIndices,:), Y(Xpointers(PairedIndices),:), d, X(UnpairedIndices,:), Y(setdiff(1:N,Xpointers(PairedIndices)),:), params);

%% Visualize the results
figure(1)
subplot(2,2,3), cla
title(sprintf('Projections:\nunpaired with Nystrom OSE'))
axis equal
axis tight
hold on
scatter([XprojPaired; XprojNystrom], [YprojPaired; YprojNystrom], 20, [t(PairedIndices)'; t(UnpairedIndices)'], 'filled')
scatter(XprojPaired, YprojPaired, 40, 0*t(PairedIndices)')
xlabel('f(x)'), ylabel('g(y)'); box on; axis equal, axis tight
legend('Paired', 'Unpaired')

figure(1)
subplot(2,2,4), cla
title(sprintf('Projections:\npaired+unpaired with NCCA'))
axis equal
axis tight
hold on
scatter([XprojPaired2; XprojUnpaired2], [YprojPaired2; YprojUnpaired2], 20, [t(PairedIndices)'; t(UnpairedIndices)'], 'filled')
scatter(XprojPaired2, YprojPaired2, 40, 0*t(PairedIndices)')
xlabel('f(x)'), ylabel('g(y)'); box on; axis equal, axis tight
legend('Paired', 'Unpaired')

