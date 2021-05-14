function [A_trains, A_tests, w_prune, mse, new_list] = prune_onestep(w, A_train, A_test, y_train, y_test, ind_list, per)
    % getindicies after pruning
    thre = quantile(abs(w), per);
    idx = abs(w) > thre;
    new_list = ind_list(idx);
    a = 1:length(w);
    list = a(idx);
    
    % get pruned matrices
%     size(A_train)
%     size(list)
    A_trains = A_train(:, list);
    A_tests = A_test(:, list);
    w_prune = min_l2(A_trains, y_train);
    mse = norm(A_tests*w_prune-y_test) / norm(y_test);
end