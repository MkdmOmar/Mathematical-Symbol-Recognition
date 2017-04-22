function mat = matCleaner(I)

[length,width] = size(I);
    for i = 0:width - 1
        if (any(I(:,width - i)))
           mat = I(:,1:(width - i));
           break;
        end
    end