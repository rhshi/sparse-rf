% This scipt is adapted from https://github.com/kirthevasank/salsa
% Please download the "salsa" file folder and add it to this working space

close all;
clear all;
clc;
addpath ../salsa/
rng('default');
warning off;

% Select dataset
% dataset = 'galaxy';       
% dataset = 'skillcraft';
% dataset = 'airfoil';
% dataset = 'CCPP';
% dataset = 'Insulin';
% dataset = 'speech';              
% dataset = 'forestfires';       
% dataset = 'housing';                  
% dataset = 'blog';       
% dataset = 'music';                    
% dataset = 'telemonitoring-total';
dataset = 'propulsion';       

% Load data
[Xtr, Ytr, Xte, Yte] = getDataset(dataset);
[nTr, numDims] = size(Xtr);
nTe = size(Xte, 1);

fprintf('Dataset: %s (n, D) = (%d, %d)\n', dataset, nTr, numDims);

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

% run SALSA
fprintf('Training with SALSA\n');
tic,
[predFunc, addOrder] = salsa(Xtr, Ytr);
toc,
YPred = predFunc(Xte);

% Print out results
predError = norm(YPred-Yte).^2/nTe;
fprintf('MSE: %0.6f\nOrder chosen by CV: %d\n', predError, addOrder);



