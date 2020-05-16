function [dataOut, info] = binarizeValidation(data, info)
% BINARIZEVALIDATION Prepara les dades per a la xarxa neuronal
%    
%   Reformata les dades per ajustar-se al format de la xarxa neuronal (20x20, binari, 1 canal)
%
    dataOut = cell(1, 2);
    
    if length(size(data)) == 3
        data = rgb2gray(data);
    end
    if not(isequal(size(data), [20 20]))
        data = imresize(data, [20 20]);
    end
    if isinteger(data)
        data = data > 0;
    end
    dataOut(1,:) = {data, info.Label};
end

