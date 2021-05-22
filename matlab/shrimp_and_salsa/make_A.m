function A = make_A(X, W)
    temp = mtimes(X, transpose(W));
    A = cat(2, cos(temp), sin(temp));
end
