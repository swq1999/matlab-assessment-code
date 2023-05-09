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
data = read(ds);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%Creating SSD Object Detection Network
inputSize = [300 300 3];

%Number of object classes to detect
classes = width(Dataset)-1;

%SSD layer
lgraph = ssdLayers(inputSize, classes, 'resnet50');

%Augment the training data
augmentedTrainingData = transform(ds,@augmentData);

%Preprocess the augmented training data to prepare for training
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

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
  [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);


%----------------------------------------- TESTING AN IMG -------------------------------------------

I = imread('338.png');
I = imresize(I,inputSize(1:2));
[person,scores] = detect(detector,I, Threshold=0.3);
I = insertObjectAnnotation(I,'rectangle',person,scores);
figure
imshow(I)

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
detectionResults = detect(detector, imds2,Threshold=0.1);

% ------------------ Average Precision -----------------------------

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, blds2);

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
[am, fppi, missRate] = evaluateDetectionMissRate(detectionResults, blds2);

figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.2f', am))






