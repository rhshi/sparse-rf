function [A_train, W, inds_track] = generate_m(X, q, N)
    d = size(X, 2);
    w_scale = 1/sqrt(q);
%     w_scale = 1
%     n = round(N / nchoosek(d, q));
    [W, Nreal, inds_track] = make_W(d, q, N, w_scale);
    A_train = make_A(X, W);
%     size(A_train)
end