function c = min_l1(A, y)
    opts = spgSetParms('verbosity',0);
    [c, ~, ~, ~] = spg_bpdn(A,y,0, opts);
end