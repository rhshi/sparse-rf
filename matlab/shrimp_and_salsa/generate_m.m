function [A_train, W, inds] = generate_m(X, q, N)
    d = size(X, 2);
    w_scale = 1/sqrt(q);
%     w_scale = 1
    n = round(N / nchoosek(d, q));
    [W, inds] = make_W(d, q, n, w_scale);

    A_train = make_A(X, W);

end