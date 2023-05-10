clc; clear;

%----------------------------------------- LOAD TRAINING DATA  -------------------------------------

myData = load('Pedestrians\train\trainLabels.mat');
names = myData.gTruth.DataSource.Source(:,1);
person =  myData.gTruth.LabelData.person;
Dataset = table(names, person);

% Shuffling train data
rng(0);
shuffledIdx = randperm(height(Dataset));
trainingData = Dataset(shuffledIdx,:);

% Image datastore
imds = imageDatastore(trainingData.names);

% label datastore
blds = boxLabelDatastore(trainingData(:,2:end));

% combine
ds = combine(imds,blds);

% Validating data
validateInputData(ds);

% Read combined data
trainData = read(ds);
I = trainData{1};
bb = trainData{2};
annImg = insertShape(I,'rectangle',bb);
annImg = imresize(annImg,2);
figure
imshow(annImg)

%Creating SSD Object Detection Network
imgInputSize = [300 300 3];

%Number of object classes to detect
classes = width(Dataset)-1;

%SSD layer
lgraph = ssdLayers(imgInputSize, classes, 'resnet50');

%Augment the training data
augmentedData = transform(ds,@augmentData);

%Preprocess the augmented training data to prepare for training
preprocessedData = transform(augmentedData,@(data)preprocessData(data,imgInputSize));
trainData = read(preprocessedData);

% SSD Object Detector options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 8, ....
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 15, ...
    'VerboseFrequency', 50, ...
    'CheckpointPath', tempdir, ...
    'Shuffle','every-epoch');

  % Training  SSD.
  [ssdDetector, info] = trainSSDObjectDetector(preprocessedData,lgraph,options);


%----------------------------------------- TESTING AN IMG -------------------------------------------

myImg = imread('338.png');
myImg = imresize(myImg,imgInputSize(1:2));
[person,scores] = detect(ssdDetector,myImg, Threshold=0.3);
myImg = insertObjectAnnotation(myImg,'rectangle',person,scores);
figure
imshow(myImg)

%------------------------------- TESTING & EVALUATING METRICS -----------------------------------

% Load test data
myData2 = load('Pedestrians\test\testLabels.mat');
names = myData2.gTruth.DataSource.Source(:,1);
person =  myData2.gTruth.LabelData.person;
Dataset2 = table(names, person);

% Image datastore
imds2 = imageDatastore('Pedestrians\test');
% label datastore
blds2 = boxLabelDatastore(Dataset2(:,'person'));

% Run Trained Detector
testResults = detect(ssdDetector, imds2,Threshold=0.1);

% ------------------ Average Precision -----------------------------

[ap,recall,precision] = evaluateDetectionPrecision(testResults, blds2);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.1f',ap))

% -------------------- Area Under Curve (AUC) ----------------------------

x = recall;
y = precision;
auc = trapz(x,y);

% ---------------------- Log Average Miss Rate ---------------------------------
[am, fppi, missRate] = evaluateDetectionMissRate(testResults, blds2);

figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.2f', am))






