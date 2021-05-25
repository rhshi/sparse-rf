function y = fn12(x)
    i = floor(length(x) / 5);
    val = 0;
    for j=1:i-1
        val = val + x(j) ^ 5;
    end
    for j=i:2*i-1
        val = val + x(j) ^ 4;
    end
    for j = 2*i:3*i-1
        val = val + x(j) ^ 3;
    end
    for j = 3*i:4*i-1
        val = val + x(j) ^ 2;
    end
    for j = 4*i:5*i-1
        val = val + x(j);
    end
    y = cos(val);
end