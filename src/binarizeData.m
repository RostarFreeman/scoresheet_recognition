function dataOut = binarizeData(dataIn)
% BINARIZEDATA Prepara les dades per a la xarxa neuronal
%    
%   Reformata les dades per ajustar-se al format de la xarxa neuronal (20x20, binari, 1 canal)
%

    dataOut = dataIn;
    if length(size(dataOut)) == 3
        dataOut = rgb2gray(dataOut);
    end
    if not(isequal(size(dataOut), [20 20]))
        dataOut = imresize(dataOut, [20 20]);
    end
    if isinteger(dataOut)
        dataOut = dataOut > 0;
    end
end

