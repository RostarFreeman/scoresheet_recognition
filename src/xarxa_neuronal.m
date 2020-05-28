close all; clearvars; clc;

trainData = datastore("..\input\RebeloDataset\Dataset", "IncludeSubfolders",true, "LabelSource","foldernames");

[trainData, valiData] = splitEachLabel(trainData, 0.85);

trainData = transform(trainData, @binarizeValidation, "IncludeInfo", true);
valiData = transform(valiData, @binarizeValidation, "IncludeInfo", true);

read(valiData)

layers = [
    imageInputLayer([20 20 1],"Name","imageinput")
    fullyConnectedLayer(256, "Name", "FirstLayer")
    batchNormalizationLayer("Name","batchnorm_1")
    leakyReluLayer("Name","relu_1")
    fullyConnectedLayer(128, "Name", "SecondLayer")
    batchNormalizationLayer("Name","batchnorm_4")
    leakyReluLayer("Name","relu_4")
    fullyConnectedLayer(length(unique(trainData.UnderlyingDatastore.Labels)),"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

plot(layerGraph(layers));

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs', 20, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valiData, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

xarxa = trainNetwork(trainData, layers, options);
labels = categories(trainData.UnderlyingDatastore.Labels);

save xarxa;
