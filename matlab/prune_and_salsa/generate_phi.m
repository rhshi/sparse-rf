function [A_train, A_test, W, inds] = generate_phi(Xtr, Xte, q, N)
    d = size(Xtr, 2);
    w_scale = 1/sqrt(q);
    n = round(N / nchoosek(d, q));
    [W, inds] = make_W(d, q, n, w_scale);

    A_train = make_A(Xtr, W);
    A_test = make_A(Xte, W);

end