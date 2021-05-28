% This scipt is adapted from https://github.com/kirthevasank/salsa
% Please download the "salsa" file folder and add it to this working space

close all;
clear all;
clc;
addpath ../salsa/
rng('default');
warning off; 

% Make data
numDims = 10;
nTr = 1000;
ratio_train = 0.7;

X = make_X(numDims, nTr);
y = zeros(nTr, 1);
for i = 1:nTr
    y(i, 1) = fn3(X(i, :));
end

Xtr = X(1:round(ratio_train*nTr)-1, :);
Xte = X(round(ratio_train*nTr):nTr, :);
Ytr = y(1:round(ratio_train*nTr)-1, 1);
Yte = y(round(ratio_train*nTr):nTr, 1);

nTe = size(Xte, 1);

fprintf('Dataset: %s (n, D) = (%d, %d)\n', "fn3", nTr, numDims);

% run pruning
fprintf('Training with M2IMP\n');
q_list = 1:5;
mse_list = zeros(1, length(q_list));
n_best_list = zeros(1, length(q_list));
i = 1;
for q = q_list
    % set up parameters
    step = 39;
    per = 0.2;
    N = 10000;
    
    % get random features embedding
    [A_train, A_test, W, inds] = generate_phi(Xtr, Xte, q, N);
    [w_len, ratio_rec, mse_rec, list_rec, ww] = prune_mse(A_train, A_test, Ytr, Yte, step, per);
    
    % get results
    min_mse = min(mse_rec);
    id = find(mse_rec==min_mse);
    n_best = w_len(id(1));
    
    mse_list(i) = min_mse;
    n_best_list(i) = n_best;
    fprintf('MSE: %0.6f with order: %d and n_best: %d\n', min_mse, q, n_best);
    i = i+1;
end

min = min(mse_list);
id = find(mse_list == min);

% print out results
fprintf('MIN MSE of purning: %0.6f\n with order: %d and n_best: %d\n',...
    min, q_list(id(1)), n_best_list(q_list(id(1))));

fprintf('Training with L2\n');
q_list = 1:5;
for q = q_list
    N = 10000;
    
    % get random features embedding
    [A_train, A_test, W, inds] = generate_phi(Xtr, Xte, q, N);
    w_l2 = min_l2(A_train, Ytr);
    
    % get results
    mse_l2 = norm(A_test*w_l2 - Yte) / norm(Yte);
    
    fprintf('MSE: %0.6f\n', mse_l2);
    i = i+1;
end

% run SALSA
fprintf('Training with SALSA\n');
tic,
[predFunc, addOrder] = salsa(Xtr, Ytr);
toc,
YPred = predFunc(Xte);

% Print out results
predError = norm(YPred-Yte).^2/nTe;
fprintf('MSE: %0.6f\nOrder chosen by CV: %d\n', predError, addOrder);



