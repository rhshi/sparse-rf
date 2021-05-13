all_data = importdata('winequality-red.csv');
data = all_data.data;
data = data(randperm(length(data)), :);

ratio_train = 0.75;
X = data(:, 1:size(data, 2)-1);
y = data(:, size(data, 2));

X_train = X(1:round(ratio_train*m)-1, :);
X_test = X(round(ratio_train*m):m, :);
y_train = y(1:round(ratio_train*m)-1);
y_test = y(round(ratio_train*m):m);

[X_train, C, S] = normalize(X_train);
X_test = normalize(X_test, 'center', C, 'scale', S);

m = length(y);

d = size(X, 2);
q = 9;
w_scale = 1/sqrt(q);
N = 10000;
n = round(N / nchoosek(d, q));

[W, inds] = make_W(d, q, n, w_scale);

A_train = make_A(X_train, W);
A_test = make_A(X_test, W);


% c_l1 = min_l1(A_train, y_train);
c_l2 = min_l2(A_train, y_train);

% plot(c_l1);
plot(c_l2);

% err_l1 = norm(A_test*c_l1-y_test) / norm(y_test);
err_l2 = norm(A_test*c_l2-y_test) / norm(y_test);

% err_l1
err_l2