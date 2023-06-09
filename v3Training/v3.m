clear; clc;

%------------------------------------ LOAD TRAINING DATA ------------------------------------------

myData = load('Pedestrians\train\trainLabels.mat');
names = myData.gTruth.DataSource.Source(:,1);
person =  myData.gTruth.LabelData.person;
Dataset = table(names, person);

% Splitting the dataset into train and test
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

% Augmenting data 
augmentedData = transform(ds, @augmentData);

% Visualize the augmented images.
myAugmentedData = cell(4,1);
for k = 1:4
    trainData = read(augmentedData);
    myAugmentedData{k} = insertShape(trainData{1,1}, 'Rectangle', trainData{1,2});
    reset(augmentedData);
end
figure
montage(myAugmentedData, 'BorderSize', 10)

% define network
imgInputSize = [227 227 3];

rng(0)
DataEstimation = transform(ds, @(data)preprocessData(data, imgInputSize));
numOfAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(DataEstimation, numOfAnchors);

% anchor boxes
myArea = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(myArea, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

% load base network
network = squeezenet;
className  = {'person'};

% Detector
yolov3Detector = yolov3ObjectDetector(network, className, anchorBoxes, ...
    'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'}, InputSize = imgInputSize);

% Preprocess training data
preprocessedData = transform(augmentedData, @(data)preprocess(yolov3Detector, data));
trainData = read(preprocessedData);

% define img
myImg = trainData{1,1};
bbox = trainData{1,2};
annotatedImg = insertShape(myImg, 'Rectangle', bbox);
annotatedImg = imresize(annotatedImg,2);
figure
imshow(annotatedImg)

reset(preprocessedData);

% Training options
epochs = 20;
batchSize = 6;
lRate = 0.001;
warmup = 700;
l2 = 0.0005;
penaltyThres = 0.5;
vel = [];

% Train model
if canUseParallelPool
   dispatchInBg = true;
else
   dispatchInBg = false;
end

mbq = minibatchqueue(preprocessedData, 2,...
        "MiniBatchSize", batchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, className), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBg,...
        "OutputCast", ["", "double"]);

% TRAINING
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);

    iter = 0;
    % my training loop.
    for ep = 1:epochs
          
        reset(mbq);
        shuffle(mbq);
        
        while(hasdata(mbq))
            iter = iter + 1;
           
            [xTrn, yTrn] = next(mbq);
            
            % Model gradients and loss
            [grdnts, state, lInfo] = dlfeval(@modelGradients, yolov3Detector, xTrn, yTrn, penaltyThres);
    
            % L2 regularization
            grdnts = dlupdate(@(g,w) g + l2*w, grdnts, yolov3Detector.Learnables);
    
            % Current learning rate
            CLR = piecewiseLearningRateWithWarmup(iter, ep, lRate, warmup, epochs);
    
            % Update detector  parameters (SGDM)
            [yolov3Detector.Learnables, vel] = sgdmupdate(yolov3Detector.Learnables, grdnts, vel, CLR);
    
            % Update state
            yolov3Detector.State = state;
              
            % Display progress.
            displayLossInfo(ep, iter, CLR, lInfo);  
                
            % Update training plot with new points.
            updatePlots(lossPlotter, learningRatePlotter, iter, CLR, lInfo.totalLoss);
        end        
    end

%----------------------------------------- TESTING AN IMG -------------------------------------------

myImg = imread('34.png');
[person,scores] = detect(yolov3Detector,myImg, Threshold=0.1);
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

% Get results
results = detect(yolov3Detector,imds2,'MiniBatchSize',6, Threshold=0.01);

% ------------------ Average Precision -----------------------------

% Evaluate AP
[ap,recall,precision] = evaluateDetectionPrecision(results,blds2);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

% -------------------- Area Under Curve (AUC) ----------------------------

x = recall;
y = precision;
auc = trapz(x,y);

% ---------------------- Log Average Miss Rate ---------------------------------

[am, fppi, missRate] = evaluateDetectionMissRate(results, blds2);

figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.2f', am))
