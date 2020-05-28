% % % % % % % % % % % % % % % %
% Projecte                    %
% Pau Torras Coloma - 1495172 %
% Marc Espinosa Gil - 1495036 %
% % % % % % % % % % % % % % % %

% L'objectiu no és, ni molt menys, ser capaç de llegir una partitura "real"
% en el sentit que aquestes poden arribar a ser extraordinariament 
% complicades i la flexibilitat del llenguatge en quant a emplaçament i 
% diversitat de fórmules equivalents juga en contra dels interessos d'un 
% algorisme generalitzat. És així que de moment ens basarem en imatges 
% sense perspectiva i ens centrarem en aquells aspectes de visió que ens 
% permetin extreure la informació exclusivament musical. Per això hem 
% generat un joc de partitures molt simples amb creixent nivell de recursos 
% musicals. Aquestes inclouen, en ordre de creixent complexitat:
%   Notes rodones
%   Notes blanques
%   Alteracions cromàtiques
%   Notes negres
%   Claus
%   Armadures
% El cas particular testbed_x00dpi.png  i custom_x00dpi.png són partitures 
% de prova amb aquests paràmetres en ment.
% Aquest projecte està inspirat en el següent paper: 
% http://vision.stanford.edu/teaching/cs231a_autumn1213_internal/project/final/writeup/distributable/sdMiller_Paper.pdf
% Durant el projecte hem anat cercant altres fonts: 
% A més de la tasca en OCR que desenvolupen alguns equips del CVC.

close all; clearvars; clc;

close all; clearvars; clc;

score_image = imread("..\input\another_300dpi.png");
score_base = 255 - rgb2gray(score_image);
score_size = size(score_base);

score_thresh = graythresh(score_base);
score_base = score_base > score_thresh;

figure, imshow(score_base, []);

% Derivada vertical de la imatge
derivada_v = conv2(score_base, [2 0 -2]', 'same') > 0.5;
derivada_v(1,:) = 0;

figure, imshow(derivada_v, []);

B = strel('line', 5, 0);                
derivada_v = imdilate(derivada_v, B);   % Minimitzar la mida dels forats de les notes

B = strel('line', 25, 0);
derivada_v = imerode(derivada_v, B);    % Eliminar agents foranis que no siguin línies

B = strel('line', 50, 0);
derivada_v = imclose(derivada_v, B);    % Reconnectar les línies

derivada_v = imtranslate(derivada_v, [0, 1], 'nearest');

derivada_v = medfilt2(derivada_v, [3,3]);

figure, imshow(derivada_v, []);
figure, imshow(cat(3,score_base*255,derivada_v*255,zeros(score_size)));

[m_votacio, theta, rho] = hough(derivada_v);

pics = houghpeaks(m_votacio, 100, ...
                  "Threshold",  .4* max(m_votacio(:)), ...
                  "NHoodSize", [3,3]);
linies_pentagrama = houghlines(derivada_v, theta, rho, pics, ...
                               "FillGap", score_size(2) / 3, ...
                               "MinLength", score_size(2) / 3);

linies_pentagrama_origen = reshape([linies_pentagrama.point1], [2, numel(linies_pentagrama)])';
linies_pentagrama_desti  = reshape([linies_pentagrama.point2], [2, numel(linies_pentagrama)])';


for i=1:numel(linies_pentagrama_origen(:,1))
    linies_pentagrama_origen(i,1) = 1;
    linies_pentagrama_desti(i,1) = score_size(2);
end

figure, imshow(score_base, []), hold on;
for i=1:numel(linies_pentagrama)
    plot([linies_pentagrama_origen(i,1), linies_pentagrama_desti(i,1)], ...
         [linies_pentagrama_origen(i,2), linies_pentagrama_desti(i,2)], ...
          "-r", ...
          "LineWidth", 2, ...
          "Marker","o");
    hold on;
end

samples = zeros([score_size(2), numel(linies_pentagrama), 2]);

for l = 1:numel(linies_pentagrama)
    for i = 1:score_size(2)
        row = linies_pentagrama_desti(l,2) - linies_pentagrama_origen(l,2);
        row = int32(linies_pentagrama_origen(l,2)) + (round(row * i / score_size(2)));
        
        if score_base(row, i)
            above = 0;
            while(row - above > 0 && score_base(row - above, i))
                above = above + 1;
            end
            samples(i,l,1) = above;
            
            below = 0;
            while(row+below <= score_size(1) && score_base(row + below, i))
                below = below + 1;
            end
            samples(i,l,2) = below;
        end
    end
end

[count_above,elems1] = histcounts(samples(:,:,1),max(samples(:,:,1),[],"all"));
[count_below,elems2] = histcounts(samples(:,:,2),max(samples(:,:,2),[],"all"));

cumsum_above = cumsum(count_above);
cumsum_below = cumsum(count_below);

percentil_above = cumsum_above ./ cumsum_above(end);
percentil_below = cumsum_below ./ cumsum_below(end);

figure, histogram(samples(:,:,1));
figure, histogram(samples(:,:,2));
lineheight_above = elems1(find(percentil_above > 0.75, 1, 'first'));
lineheight_below = elems2(find(percentil_below > 0.75, 1, 'first'));

score_nocleff = score_base;
for l=1:numel(linies_pentagrama)
    for i=1:score_size(2)
        row = linies_pentagrama_desti(l,2) - linies_pentagrama_origen(l,2);
        row = int32(linies_pentagrama_origen(l,2)) + (round(row * i / score_size(2)));
        
        for height=row-lineheight_below:row+lineheight_above
            score_nocleff(height, i) = 0;
        end
    end
end

figure, imshow(score_nocleff, []);

B = strel("square", lineheight_below + lineheight_above + 2);
score_nocleff = imclose(score_nocleff, B);          % Closing de la imatge per tancar els forats resultants d'eliminar pentagrama
score_nocleff = and(score_nocleff, score_base);     % And per garantir que no s'afegeixen nous pixels

% B = strel("square", 3);
% score_nocleff = imopen(score_nocleff, B);


imwrite(score_nocleff, "../output/segmentation/binarized.png");
figure, imshow(score_nocleff, []);

[sarray, sindex] = sort(linies_pentagrama_origen(:,2));
linies_pentagrama_origen = linies_pentagrama_origen(sindex,:);
linies_pentagrama_desti = linies_pentagrama_desti(sindex,:);

pentagrames = zeros(floor(numel(linies_pentagrama)/5), 2);
root_pentagrames = zeros([floor(numel(linies_pentagrama)/5) 1]);

height_pentagrames = median([linies_pentagrama_origen(:,2)' 0] - [0 linies_pentagrama_origen(:,2)']);

% Primer trobar les primeres i ultimes linies del pentagrama
for i=1:floor(numel(linies_pentagrama)/5)
    pentagrames(i,1) = min(linies_pentagrama_origen(((i-1)*5)+1, 2), linies_pentagrama_desti(((i-1)*5)+1,2));
    pentagrames(i,2) = max(linies_pentagrama_origen(i*5, 2), linies_pentagrama_desti(i*5,2));
    root_pentagrames(i) = pentagrames(i,2);
end

% Trobar la minima diferencia entre linies de pentagrames diferents
mindif = inf;
for i = 1:size(pentagrames)-1
    if pentagrames(i,2) - pentagrames(i+1,1) < mindif
        mindif = pentagrames(i+1,1) - pentagrames(i,2);
    end
end

% Redefinir els limits dels pentagrames
difheight = floor(mindif/2);
for i = 1:size(pentagrames)
    pentagrames(i,1) = pentagrames(i,1) - difheight + 1;
    pentagrames(i,2) = pentagrames(i,2) + difheight - 1;
end

% Esborrar qualsevol cosa fora dels pentagrames
score_nocleff(1:pentagrames(1,1),:) = 0;
for i=1:size(pentagrames) - 1
    score_nocleff(pentagrames(i,2):pentagrames(i+1,1),:) = 0;
end
score_nocleff(pentagrames(end,2):end,:) = 0;

figure, imshow(score_nocleff), hold on;
for i = 1:size(pentagrames)
    plot([1, score_size(2)], [pentagrames(i,1) pentagrames(i,1)], '-r');
    plot([1, score_size(2)], [pentagrames(i,2) pentagrames(i,2)], '-g');
    hold on;
end

% Carregar la xarxa entrenada amb el dataset
load xarxa.mat;

% Trobar les components connexes
components_connexes = regionprops(score_nocleff, 'BoundingBox');
score_elms = zeros([20, 20, length(components_connexes)], "logical");
score_labels = [];

% Treure les imatges de cada element
figure(); imshow(score_image, []), hold on;
for ii=1:length(components_connexes)
    boundbox = int32(components_connexes(ii).BoundingBox);
    rectangle("Position", boundbox, "EdgeColor", "b"), hold on;
    score_elms(:,:,ii) = not(imresize(score_nocleff(boundbox(2):boundbox(2) + boundbox(4), boundbox(1):boundbox(1) + boundbox(3)), [20 20]));
    score_labels = [score_labels classify(xarxa, reshape(score_elms(:,:,ii), [20 20 1]))];
    text(double(boundbox(1) - 5), double(boundbox(2) - 5), score_labels(end),"Color","r");
end

output = fopen("../output/output_score.txt", "wt");

elm_coords = reshape(cell2mat(struct2cell(components_connexes))', [4 length(components_connexes)]);
loy = elm_coords(2,:);
hiy = loy + elm_coords(4,:);

% Versio sense linies de les notes
erased_note_lines = score_nocleff;
B = strel("square", 5);
erased_note_lines = imerode(erased_note_lines, B);
B = strel("square", 7);

erased_note_lines = imdilate(erased_note_lines, B);
erased_note_lines = and(erased_note_lines, score_nocleff);

for ii=1:length(pentagrames)
    belong = and(loy >= pentagrames(ii,1), hiy <= pentagrames(ii,2));
    elms_in_pentagram = elm_coords(:,belong);
    lox = elm_coords(1,belong);
    [lox, order] = sort(lox);
    printlabels = score_labels(belong);
    
    notes = or(or(or(or(or(printlabels == "Notes", printlabels == "NotesOpen"), printlabels == "quaver"), ...
        printlabels == "SemiBreve"), printlabels == "semiquaver"), printlabels == "quaver");

    printlabels = printlabels(order);
    
    % Calcular la diferencia d'altura entre la nota i la línia més baixa
    % del pentagrama en qüestió.
    
    for jj=1:length(printlabels)
        fprintf(output, "%s", printlabels(jj));
        if notes(jj)
            center = int32([0.0 0.0]);
            blackpx = 0;
            for zz = int32(elms_in_pentagram(1,jj):elms_in_pentagram(1,jj)+elms_in_pentagram(3,jj))
                for kk = int32(elms_in_pentagram(2,jj):elms_in_pentagram(2,jj)+elms_in_pentagram(4,jj))
                    if erased_note_lines(kk,zz)
                        center = center + [zz*int32(erased_note_lines(kk,zz)), kk*int32(erased_note_lines(kk,zz))];
                        blackpx = blackpx + 1;
                    end
                end
            end
            center = center / blackpx;
            scatter(center(1), center(2)), hold on;
            note = int32(int32(root_pentagrames(ii) - center(2)) / floor(height_pentagrames/2));
            
            text(double(center(1)) + 15, double(center(2)) + 10, string(note), "Color", "g"), hold on;
            
            fprintf(output, "(%i)", note);
        end
        fprintf(output, "\n");
    end
end
print('-dpng', '-r600', "../output/output_score.png");
saveas(gcf,"../output/output_score.png");

% TODO: agafar el resultat i passar-lo per un parser i generar el MIDI corresponent. La gramàtica és la següent:
% <Tema> ::= {<Compas>} BarLines
% <Compas> ::= BarLines {<Símbol>}
% <Símbol> ::= <Nota> | <Silenci> | <Armadura> | <Clau> | <Indicació de Compàs>
% <Nota> ::= [<Alteració>] (Notes | NotesOpen | quaver | SemiBreve | semiquaver) [Dots]
% <Silenci> ::= Rests | Rest-doublewhole | Rest-half-whole | Rest-quaver
% <Armadura> ::= <Alteració> {<Alteració>}
% <Alteració> ::= Sharps | Naturals | Flat
% Problema:
% És ambigua a causa de la presència d'armadures i alteracions accidentals associades a notes