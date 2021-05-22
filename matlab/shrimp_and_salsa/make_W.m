function [W, Nreal] = make_W(d, q, N, scale)
    num_supports = nchoosek(d, q);
    if num_supports > N
        Nreal = N;
        W = zeros(Nreal, d);
        for i = 1:Nreal
            w = rand(1, d) * scale;
            inds = randsample(d, d-q);
            w(inds) = 0;
            W(i, :) = w;
        end
    else
        n = round(N / num_supports);
        Nreal = n * num_supports;
        inds = nchoosek(1:d, d-q);
        W = zeros(n*num_supports, d);
        for i = 1:num_supports
            ind = inds(i, :);
            for j = 1:n
                w = randn(1, d) * scale;
                w(ind) = 0;
                W((i-1)*n+j, :) = w;
            end
        end
    end
end
