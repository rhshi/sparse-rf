close all;
clear all;
clc;


% data_list = {'propulsion', 'galaxy', 'airfoil', 'speech', 'CCPP', ...
%     'forestfires', 'housing', 'music', 'Insulin', 'skillcraft', ...
%     'telemonitoring-total', 'blog'};

data_list = {'speech'};

% data_list = {'propulsion'};
% per_list =  [0.1, 0.15, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.55, 0.6];
per_list = [0.2, 0.3, 0.4];
mse_dict = robust_test(data_list, per_list);

% tiledlayout(3,4)
% for i = 1:length(data_list)
%     nexttile
%     err = mse_dict(data_list{i});
%     scatter(per_list, err, 'blue')
%     scatter(per_list, err, 'blue')
%     title(data_list{i})
%     xlabel("pruning percentage")
%     ylabel("Test MSE")
% end

% tiledlayout(3,4)
% for i = 1:length(data_list)
%     nexttile
%     qlist = q_dict(data_list{i});
%     scatter(per_list, qlist, 'green')
%     scatter(per_list, qlist, 'green')
%     title(data_list{i})
%     xlabel("pruning percentage")
%     ylabel("best q order")
% end

save("./results/robust_test_mse_fulld", 'mse_dict');
save("./results/robust_test_q_fulld", 'q_dict');