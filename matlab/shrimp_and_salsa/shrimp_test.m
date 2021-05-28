close all;
clear all;
clc;


data_list = {'fn5'};
d = 100;
qs = [1];
m = 2000;
seeds = [55, 56, 57];
per =  0.2;
for i = 1:length(data_list)
    params = struct();
    params.numPartsKFoldCV = 10;
    params.orderCands = qs(i);
    params.step = 50;
    params.N = 10000;
    params.per = per;
    dataset = data_list{i};
    
    prune_errs = [];
    N_bests = [];
    l1_errs = [];
    l1_sparse = [];
    l2_errs = [];
    K_errs = [];
    
    for j = 1:length(seeds)
        [Xtr, Ytr, Xte, Yte] = getDataset(dataset, seeds(j), d, m);
        [nTr, numDims] = size(Xtr);
        nTe = size(Xte, 1);
        fprintf('Dataset: %s (n, D) = (%d, %d), seed: %d\n', dataset, nTr, numDims, seeds(j));
    
    
        tic,
        best_model = shrimp(Xtr, Ytr, params);
        %--test on test datasets
        A_train = make_A(Xtr, best_model("W"));
        A_test = make_A(Xte, best_model("W"));
        A_test_prune = A_test(:, best_model("id_list"));
        y_pred = A_test_prune*best_model("w");
        pred_error = norm(y_pred - Yte).^2/nTe;
        prune_errs = [prune_errs, pred_error];
        N_bests = [N_bests, best_model("n_best")];

        c_l1 = min_l1(A_train, Ytr);
        y_pred = A_test*c_l1;
        pred_error = norm(y_pred - Yte).^2/nTe;
        l1_errs = [l1_errs, pred_error];
        l1_sparse = [l1_sparse, length(nonzeros(c_l1))];

        c_l2 = min_l2(A_train, Ytr);
        y_pred = A_test*c_l2;
        pred_error = norm(y_pred - Yte).^2/nTe;
        l2_errs = [l2_errs, pred_error];

        bws = sqrt(qs(i)) * ones(d); 
        [K, ~] = espKernels(Xtr, Xtr, bws, qs(i));
        Ktr = K / nchoosek(d, qs(i));
        [K, ~] = espKernels(Xte, Xtr, bws, qs(i));
        Kte = K / nchoosek(d, qs(i));
        c_K = min_l2(Ktr, Ytr);
        y_pred = Kte*c_K;
        pred_error = norm(y_pred - Yte).^2/nTe;
        K_errs = [K_errs, pred_error];
    end
    fprintf('SHRIMP MSE: %.2e with n_bests: %d %d %d, averaged: %f\n',...
        mean(prune_errs), N_bests, mean(N_bests));
    fprintf('L1 MSE: %.2e with sparsity: %d %d %d, averaged %f\n', mean(l1_errs), l1_sparse, mean(l1_sparse));
    fprintf('L2 MSE: %.2e\n', mean(l2_errs));
    fprintf('Kernel MSE: %.2e\n', mean(K_errs));
    
end