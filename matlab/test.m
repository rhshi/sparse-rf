d = 5;
q = 1;
ratio_train = 0.7;
w_scale = 1/sqrt(q);
N = 10000;
n = round(N / nchoosek(d, q));
m = 1000;

X = make_X(d, m);
y = zeros(m, 1);

for i = 1:m
    y(i, 1) = fn2(X(i, :));
end
    
X_train = X(1:round(ratio_train*m)-1, :);
X_test = X(round(ratio_train*m):m, :);
y_train = y(1:round(ratio_train*m)-1, 1);
y_test = y(round(ratio_train*m):m, 1);

[W, inds] = make_W(d, q, n, w_scale);

A_train = make_A(X_train, W);
A_test = make_A(X_test, W);

group = [];
for i = 1:2*length(inds)
    for j = 1:n
        group = [group, i];
    end
end

[c_bomp, residual] = BOMP(A_train, y_train, group, 5);
c_l2 = min_l2(A_train, y_train);
norm(A_test* c_bomp - y_test) / norm(y_test)
norm(A_test* c_l2 - y_test) / norm(y_test)