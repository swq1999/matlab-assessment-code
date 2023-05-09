clear; clc;

%------------------------------------ LOAD TRAINING DATA ------------------------------------------

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

% Minimum Input Size
inputSize = [224 224 3];
classes = width(Dataset)-1;
trainingDataForEstimation = transform(ds,@(data)preprocessData(data,inputSize));
anchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, anchors);
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
lgraph = yolov2Layers(inputSize,classes,anchorBoxes,featureExtractionNetwork,featureLayer);
augmentedTrainingData = transform(ds,@augmentData);

% Augmenting & Visualizing Images
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'rectangle',data{2});
    reset(augmentedTrainingData);
end

figure
montage(augmentedData,'BorderSize',10)
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Training options
options = trainingOptions('adam', ...
        'MiniBatchSize', 6, ....
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20);
  
  % Train YOLov2
  [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);

%----------------------------------------- TESTING AN IMG -------------------------------------------

I = imread('34.png');
I = imresize(I,inputSize(1:2));
[person,scores] = detect(detector,I, Threshold=0.6);
I = insertObjectAnnotation(I,'rectangle',person,scores);
figure
imshow(I)

%------------------------------- TESTING & EVALUATING METRICS -----------------------------------

% Load test dataset
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






