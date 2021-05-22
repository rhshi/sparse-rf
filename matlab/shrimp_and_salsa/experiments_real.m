% This scipt is adapted from https://github.com/kirthevasank/salsa
% Please download the "salsa" file folder and add it to this working space

close all;
clear all;
clc;
addpath ./salsa/
rng('default');
warning off;

% Select dataset
dataset = 'propulsion';
% dataset = 'galaxy';
% dataset = 'airfoil';
% dataset = 'speech'; 
% dataset = 'CCPP';
% dataset = 'forestfires';
% dataset = 'housing';
% dataset = 'music'; 
% dataset = 'Insulin';
% dataset = 'skillcraft'; 
% dataset = 'telemonitoring-total';
% dataset = 'blog';
                
                    
% Load data
[Xtr, Ytr, Xte, Yte] = getDataset(dataset);
[nTr, numDims] = size(Xtr);
nTe = size(Xte, 1);
fprintf('Dataset: %s (n, D) = (%d, %d)\n', dataset, nTr, numDims);

% run shrimp
fprintf('Training with SHRIMP\n');

%--setup parameters
params = struct();
params.per = 0.5;
params.numPartsKFoldCV = 10;
params.orderCands = 1:10;
params.step = 50;
params.N = 10000;


tic,
best_model = shrimp(Xtr, Ytr, params);
%--test on test datasets
A_test = make_A(Xte, best_model("W"));
A_test_prune = A_test(:, best_model("id_list"));
y_pred = A_test_prune*best_model("w");
pred_error = norm(y_pred - Yte).^2/nTe;
fprintf('SHRIMP MSE: %.2e with pruning rate %.2f, Order chosen by Validation: %d with n_best: %d\n\n',...
    pred_error, params.per, best_model("q"), best_model("n_best"));
toc,

% run SALSA
fprintf('Training with SALSA\n');
tic,
[predFunc, addOrder] = salsa(Xtr, Ytr);
YPred = predFunc(Xte);

% Print out results
predError = norm(YPred-Yte).^2/nTe;
fprintf('SALSA MSE: %.2e, Order chosen by CV: %d\n', predError, addOrder);
toc,



