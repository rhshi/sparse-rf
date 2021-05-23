close all;
clear all;
clc;


% data_list = {'propulsion', 'galaxy', 'airfoil', 'speech', 'CCPP', ...
%     'forestfires', 'housing', 'music', 'Insulin', 'skillcraft', ...
%     'telemonitoring-total', 'blog'};

data_list = {'propulsion'};
per_list =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
mse_dict = robust_test(data_list, per_list);