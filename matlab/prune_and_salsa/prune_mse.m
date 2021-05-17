function [w_len, ratio_record, mse_record, list_rec, ww]= prune_mse(A_train, A_test, y_train, y_test, step, per)
    w_prune = min_l2(A_train, y_train);
    y_preds = A_test*w_prune;
    ratio = norm(y_preds-y_test) / norm(y_test);
    mse = norm(y_preds-y_test).^2/length(y_test);
    
    ratio_record = zeros(1, step+1);
    ratio_record(1) = ratio;
    mse_record = zeros(1, step+1);
    mse_record(1) = mse;
    
    w_len = zeros(1, step+1);
    w_len(1) = length(w_prune);
    
    ww = containers.Map;
    ww(int2str(length(w_prune))) = w_prune;
    
    ind_list = 1:length(w_prune);
    list_rec = containers.Map;
    list_rec(int2str(length(w_prune))) = ind_list;
    
    for i = 1:step
        [A_train, A_test, w_prune, ratio, mse, ind_list] = prune_os_mse(w_prune, A_train, A_test, y_train, y_test, ind_list, per);
        w_len(i+1) = length(w_prune);
        ratio_record(i+1) = ratio;
        mse_record(i+1) = mse;
        list_rec(int2str(length(w_prune))) = ind_list;
        ww(int2str(length(w_prune))) = w_prune;   
    end
    
end

function [A_trains, A_tests, w_prune, ratio, mse, new_list] = prune_os_mse(w, A_train, A_test, y_train, y_test, ind_list, per)
    % getindicies after pruning
    thre = quantile(abs(w), per);
    idx = abs(w) > thre;
    new_list = ind_list(idx);
    a = 1:length(w);
    list = a(idx);
    
    % get pruned matrices
    A_trains = A_train(:, list);
    A_tests = A_test(:, list);
    w_prune = min_l2(A_trains, y_train);
    y_preds = A_tests*w_prune;
    ratio = norm(y_preds-y_test) / norm(y_test);
    mse = norm(y_preds-y_test).^2/length(y_test);
end