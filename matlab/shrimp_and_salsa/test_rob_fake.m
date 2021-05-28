close all;
clear all;
clc;


data_list = {'ishigami'};
d = 10;
per_list =  [0.2];
mse_dict = robust_test(data_list, per_list, d);

tiledlayout(length(data_list),1)
for i = 1:length(data_list)
    nexttile
    plot(per_list, mse_dict(data_list{i}))
    title(data_list{i})
    xlabel("pruning percentage")
    ylabel("Test MSE")
end

save(".results/robust_test_fake", 'mse_dict');