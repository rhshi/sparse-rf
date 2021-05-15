data = importdata('airfoil_self_noise.dat');
data = data(randperm(length(data)), :);

ratio_train = 0.75;
X = data(:, 1:size(data, 2)-1);
y = data(:, size(data, 2));
m = size(data, 1);

X_train = X(1:round(ratio_train*m)-1, :);
X_test = X(round(ratio_train*m):m, :);
y_train = y(1:round(ratio_train*m)-1);
y_test = y(round(ratio_train*m):m);

[X_train, C, S] = normalize(X_train);
X_test = normalize(X_test, 'center', C, 'scale', S);

m = length(y);

d = size(X, 2);
q = 3;
w_scale = 1/sqrt(q);
N = 10000;
n = round(N / nchoosek(d, q));

[W, inds] = make_W(d, q, n, w_scale);

A_train = make_A(X_train, W);
A_test = make_A(X_test, W);

group = [];
for i = 1:length(inds)
    for j = 1:n
        group = [group, i];
    end
end
group = [group, group];



tiledlayout(5,1)

% l2
c_l2 = min_l2(A_train, y_train);
errl2 = norm(A_test*c_l2-y_test) / norm(y_test)
nexttile
scatter(1:length(c_l2), c_l2, 2)
title(errl2 + " naca l2")

% BOMP
c_bomp = BOMP(A_train, y_train, group, 4);
errbomp = norm(A_test*c_bomp-y_test) / norm(y_test)
nexttile
scatter(1:length(c_bomp), c_bomp, 2)
title(errbomp + " naca bomp")

%l1
c_l1 = min_l1(A_train, y_train);
errl1 = norm(A_test*c_l1-y_test) / norm(y_test)
nexttile
scatter(1:length(c_l1), c_l1, 2)
title(errl1 + " naca l1")

% pruning
step = 39;
per = 0.2;
[w_len, mse_rec, list_rec, ww] = prune_total(A_train, A_test, y_train, y_test, step, per);

min_mse = min(mse_rec)

id = find(mse_rec==min_mse);
n_best = w_len(id);
nexttile
scatter(list_rec(int2str(n_best)), ww(int2str(n_best)), 2)
title("naca pruning: " + min_mse + ", nbest: " + n_best)

nexttile
loglog(w_len, mse_rec)
hold on
scatter(n_best, min_mse)
title("naca mse ratio of pruning")
hold off



