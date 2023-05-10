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
ImgInputSize = [224 224 3];
numOfClasses = width(Dataset)-1;
estimationData = transform(ds,@(data)preprocessData(data,ImgInputSize));
anchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(estimationData, anchors);
fNetwork = resnet50;
fLayer = 'activation_40_relu';
lgraph = yolov2Layers(ImgInputSize,numOfClasses,anchorBoxes,fNetwork,fLayer);
augmentedTrainData = transform(ds,@augmentData);

% Augmenting & Visualizing Images
myAugmentedData = cell(4,1);
for k = 1:4
    trainData = read(augmentedTrainData);
    myAugmentedData{k} = insertShape(trainData{1},'rectangle',trainData{2});
    reset(augmentedTrainData);
end

figure
montage(myAugmentedData,'BorderSize',10)
preprocessedData = transform(augmentedTrainData,@(data)preprocessData(data,ImgInputSize));
trainData = read(preprocessedData);
myImg = trainData{1};
mybbox = trainData{2};
annotatedImage = insertShape(myImg,'rectangle',mybbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Training options
options = trainingOptions('adam', ...
        'MiniBatchSize', 6, ....
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20);
  
  % Train YOLov2
  [YOLOv2Detector,info] = trainYOLOv2ObjectDetector(preprocessedData,lgraph,options);

%----------------------------------------- TESTING AN IMG -------------------------------------------

myImg = imread('rider.jpg');
myImg = imresize(myImg,ImgInputSize(1:2));
[person,scores] = detect(YOLOv2Detector,myImg, Threshold=0.6);
myImg = insertObjectAnnotation(myImg,'rectangle',person,scores);
figure
imshow(myImg)

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
myDetectionResults = detect(YOLOv2Detector, imds2,Threshold=0.1);

% ------------------ Average Precision -----------------------------

[ap,recall,precision] = evaluateDetectionPrecision(myDetectionResults, blds2);

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

[am, fppi, missRate] = evaluateDetectionMissRate(myDetectionResults, blds2);

figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.2f', am))






