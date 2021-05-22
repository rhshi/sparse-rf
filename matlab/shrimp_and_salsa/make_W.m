function [W, inds_track] = make_W(d, q, n, scale)
    num_supports = nchoosek(d, q);
    inds = nchoosek(1:d, d-q);
    W = zeros(n*num_supports, d);
    inds_track = zeros(num_supports, q);
    for i = 1:num_supports
        ind = inds(i, :);
        inds_track(i, :) = setdiff(1:d, ind);
        for j = 1:n
            w = randn(1, d) * scale;
            w(ind) = 0;
            W((i-1)*n+j, :) = w;
        end
    end
end
