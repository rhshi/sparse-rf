function [w_len, mse_record, list_rec, ww]= prune_total(A_train, A_test, y_train, y_test, step, per)
    w_prune = min_l2(A_train, y_train);
    mse = norm(A_test*w_prune-y_test) / norm(y_test);
    
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
        [A_train, A_test, w_prune, mse, ind_list] = prune_onestep(w_prune, A_train, A_test, y_train, y_test, ind_list, per);
        
%         i
%         length(w_prune)
        % record results
        w_len(i+1) = length(w_prune);
        mse_record(i+1) = mse;
        list_rec(int2str(length(w_prune))) = ind_list;
        ww(int2str(length(w_prune))) = w_prune;   
    end
    
end