function X = make_X(d, m, scale)
    if nargin == 2
        X = 2*rand(m, d)-1;
    else
        X = 2*scale*rand(m, d)-scale;
    end
	
end

