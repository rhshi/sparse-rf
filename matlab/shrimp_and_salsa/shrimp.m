function best_model = shrimp(X, Y, params)
% This function implements SHRIMP: Sparse Random Feature Models with Iterative Magnitude Pruning.
% Refer the paper for more details.
% Inputs:
%   (X, Y): The training data and labels
%   params: A structure which optionally contain the following hyper-parameters
%     - numPartsKFoldCV: Number of partitions for K-Fold cross validation (CV).
%     - numTrialsKFoldCV: Number of partitions to average the CV error over.
%     - lambdaRange: A range within which the candidates for lambda will be selected
%         with logarithmic spacing.
%     - orderCands: Candidate values for the additive order.
% Outputs:
%   predFunc: A function handle which can be used to estimate the function for new
%             points.
%   addOrder: The order of the additive model chosen.
%   We cross validate to choose the low order q and the resulting n_pruned.

  % prelims
  m = size(X, 1);

  % shuffle Data
  shuffleOrder = randperm(m);
  X = X(shuffleOrder, :);
  Y = Y(shuffleOrder, :);

  % Params for CV
  if ~exist('params', 'var') || isempty(params)
    params = struct();
  end
  
  if ~isfield(params, 'numPartsKFoldCV')
    params.numPartsKFoldCV = 10;
  end
  
  if ~isfield(params, 'orderCands')
    params.orderCands = 1:5;
  end
  
  if ~isfield(params, 'N')
    params.N = 10000;
  end
  
  if ~isfield(params, 'step')
    params.step = 40;
  end
  
  if ~isfield(params, 'per')
    params.per = 0.2;
  end
  
  % Copy over to workspace
  orderCands = params.orderCands;
  numOrderCands = numel(params.orderCands);


  % Now for each order determine the best N_prune and validation error
  best_model = containers.Map;
  best_model("n_best") = -1;
  best_model("id_list") = -1;
  best_model("min_val") = inf;
  best_model("W") = -1;
  best_model("q") = -1;
  best_model("A_train") = 0;
  
  for i = 1:numOrderCands
    q = orderCands(i);
    fprintf('Order: %d, ', q);
    [A_train, W, ] = generate_m(X, q, params.N);
%     size(A_train)
    [n_best, id_list, min_mse, w] = validate(A_train, Y, params.numPartsKFoldCV, params.step, params.per);
    if min_mse < best_model("min_val")
        best_model("n_best") = n_best;
        best_model("id_list") = id_list;
        best_model("min_val") = min_mse;
        best_model("W") = W;
        best_model("q") = q;
        best_model("w") = w;
%         best_model("A_train") = A_train;
    end      
  end
  
%   A = best_model("A_train");
%   A_train_final = A(:, best_model("id_list"));
%   w = min_l2(A_train_final, Y);
%   best_model("A_train") = 0;
  
%   best_model("w") = w;

end


% Validation part
function [n_best, id_list, min_mse, w] = validate(X, Y, numPartsKFoldCV, step, per)
%   validErr = 0;
  m = size(X, 1);
  cvIter = 1;
  testStartIdx = round( (cvIter-1)*m/numPartsKFoldCV + 1);
  testEndIdx = round( cvIter*m/numPartsKFoldCV );
  trainIdxs = [1:(testStartIdx-1), (testEndIdx+1):m]';
  testIdxs = [testStartIdx:testEndIdx]';
  nVal = testEndIdx - testStartIdx + 1;
  nTr = m - nVal;
  Atr = X(trainIdxs, :);
  Aval = X(testIdxs, :);
  Ytr = Y(trainIdxs, :);
  Yval = Y(testIdxs, :);

  [w_len, mse_rec, list_rec, ww] = shrimp_prune(Atr, Aval, Ytr, Yval, step, per);
  
   % get results
%    w_len
%    mse_rec
   min_mse = min(mse_rec);
   id = find(mse_rec==min_mse);
   n_best = w_len(id(1));
   id_list = list_rec(int2str(n_best));
   w = ww(int2str(n_best));
    
   fprintf('Valid-Err MSE: %.2e, n_best: %d\n', min_mse, n_best);
    
end


function [w_len, mse_record, list_rec, ww] = shrimp_prune(A_train, A_test, y_train, y_test, step, per)
    w_prune = min_l2(A_train, y_train);
    y_preds = A_test*w_prune;
    
    mse = norm(y_preds-y_test).^2/length(y_test);
    mse_record = zeros(1, step+1);
    mse_record(1) = mse;
    
    w_len = -1 * ones(1, step+1);
    w_len(1) = length(w_prune);
    
    ind_list = 1:length(w_prune);
    list_rec = containers.Map;
    list_rec(int2str(length(w_prune))) = ind_list;
    
    % record ww
    ww = containers.Map;
    ww(int2str(length(w_prune))) = w_prune;
    
    for i = 1:step
        [A_train, A_test, w_prune, mse, ind_list] = prune_os(w_prune, A_train, A_test, y_train, y_test, ind_list, per);
        w_len(i+1) = length(w_prune);
        mse_record(i+1) = mse;
        list_rec(int2str(length(w_prune))) = ind_list;
        ww(int2str(length(w_prune))) = w_prune; 
    end
    
end

function [A_trains, A_tests, w_prune, mse, new_list] = prune_os(w, A_train, A_test, y_train, y_test, ind_list, per)
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
    mse = norm(y_preds-y_test).^2/length(y_test);
end



