function [mse_dict, q_dict] = robust_test(data_list, per_list)

mse_dict = containers.Map;
q_dict = containers.Map;
for i = 1:length(data_list)
    dataset = data_list{i};
    if nargin == 2
        [Xtr, Ytr, Xte, Yte] = getDataset(dataset);
    else 
        [Xtr, Ytr, Xte, Yte] = getDataset(dataset, d);
    end
    [nTr, numDims] = size(Xtr);
    nTe = size(Xte, 1);
    mse_list = -1 * ones(1, length(per_list));
    qbest_list = -1 * ones(1, length(per_list));
    fprintf('Dataset: %s (n, D) = (%d, %d)\n', dataset, nTr, numDims);
    
    %--setup parameters
    params = struct();
    params.numPartsKFoldCV = 10;
    params.orderCands = 1:numDims;
    params.step = 100;
    params.N = 10000;
    
    for j = 1:length(per_list)
        params.per = per_list(j);
        tic,
        best_model = shrimp(Xtr, Ytr, params);
        %--test on test datasets
        A_test = make_A(Xte, best_model("W"));
        A_test_prune = A_test(:, best_model("id_list"));
        y_pred = A_test_prune*best_model("w");
        pred_error = norm(y_pred - Yte).^2/nTe;
        fprintf('SHRIMP MSE: %.2e with pruning rate %.2f;\n Order chosen by Validation: %d with n_best: %d\n\n',...
        pred_error, params.per, best_model("q"), best_model("n_best"));
        mse_list(j) = pred_error; 
        qbest_list(j) = best_model("q");
        toc, 
    end
    
    mse_dict(dataset) = mse_list;
    q_dict(dataset) = qbest_list;
end

end

