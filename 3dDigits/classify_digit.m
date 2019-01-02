% Project script
% This script contain the whole project process from loading the data,
% preprocess the data, split the data in training a testing subsets, PCA
% analysis and train all the models we decide to experiment with. In
% addition, it runs cross-validation to two selected algorithms that
% performed according our expectations.


%==========================LOADING DATA TO WORKSPACE=====================
tic;
digitsData=preprocessing_data;
time_preprocessing=toc/60;
labels=[];
for i=0:9
	labels=[labels;repmat(i,100,1)];
end
vLabel=zeros(size(labels,1),10);
for i=1:size(labels,1)
	vLabel(i,labels(i)+1)=1;
end
%==========================SPLIT DATASET=================================
p=0.85; % training proportion to split the dataset
randomIndexes=randperm(1000); %mix the samples randomly
dataTraining=digitsData(randomIndexes(1:p*1000),:); %first p*N randomly mix samples
labelsTraining=labels(randomIndexes(1:p*1000)); %first p*N randomly mix samples
vLabelTraining=vLabel(randomIndexes(1:p*1000),:); %first p*N randomly mix samples
dataTesting=digitsData(randomIndexes(p*1000+1:end),:); %the rest randomly mix samples
labelsTesting=labels(randomIndexes(p*1000+1:end)); %the rest randomly mix samples
vLabelTesting=vLabel(randomIndexes(p*1000+1:end),:); %the rest randomly mix samples
%=======================PCA ANALYISIS AND TRANSFORMATION=================
% PCA: the number of features is too high so we want to find the
% eigenVector which contain the majority of the variance and reduce the
% dimensionality of our classification problem. We are going to choose the
% vector that explain a cumulate variance of 90%.
explainedVar=0.9;
tfTrainPC=pcaTransform(dataTraining, explainedVar, dataTraining);
tfTestPC=pcaTransform(dataTraining, explainedVar, dataTesting);

%========================TRAINING AND TESTING MODEL======================
% K-NN: K NEAREST NEIGHBORS----------------------------------------------
% Train a model with k neirest neighbors classifier. We test k from 2
% to 9 with better results with 2 nearest neighbors, however we found it
% overfits the data having poor results with the test data.
k=2;
tic;
predClassKNN=knn(labelsTraining',tfTrainPC',tfTrainPC',k);
time_knn=toc/60;
acc_knn=mean(predClassKNN'==labelsTraining);
tic;
predClassTKNN=knn(labelsTraining',tfTrainPC',tfTestPC',k);
time_knnT=toc/60;
acc_knnT=mean(predClassTKNN'==labelsTesting);
[acc_knn,acc_knnT]

% LINEAR CLASSIFIER------------------------------------------------------
% We test LMS to continue our journey experimenting with different
% classification models. LMS particularly works better with higher levels
% during the preprocessing process, in our test 40 levels works better.
% The results were still poor, however we were expecting this type of
% result from the linear classifier.
tic;
[wLms,windLms] = lms(labelsTraining', tfTrainPC');
prClassLMS=predictLMS(tfTrainPC',wLms,windLms);
prClassTLMS=predictLMS(tfTestPC',wLms,windLms);
time_lms=toc/60;
acc_lms=mean(prClassLMS'==labelsTraining);
acc_lmsT=mean(prClassTLMS'==labelsTesting);
[acc_lms,acc_lmsT]

% NEURAL NETWORK1: LOGISTIC ACTIVATION FUNCTION--------------------------
% We test a NN with three layers: inputs, hidden and
% outputs. In the hidden layer we set 25 neurons, we also test other
% numbers of neurons finding 25 the best suitable. In addition we determine
% the learning rate as 0.01, 100000 iterations and logistic activation
% function. The final results show us that we are overfitting having higher
% than 93% of accuracy for training data but 0.56-0.6 for the test set.
tic;
[wLayerHid,wLayerOut]=nnLog(tfTrainPC', vLabelTraining', 25);
predClassLog=predictLog(tfTrainPC',wLayerHid,wLayerOut);
predClassLogT=predictLog(tfTestPC',wLayerHid,wLayerOut);
time_nnLog=toc/60;
acc_nnLog=mean(predClassLog'==labelsTraining);
acc_nnLogT=mean(predClassLogT'==labelsTesting);
[acc_nnLog, acc_nnLogT]

% NEURAL NETWORK2: SIGMOID (TANH) ACTIVATION FUNCTION--------------------
% We test a NN with three layers: inputs, hidden and
% outputs. In the hidden layer we set 25 neurons, we also test other
% numbers of neurons finding 25 the best suitable. In addition we determine
% the learning rate as 0.001, 200000 iterations and sigmoid activation
% function. The final results show us that the sigmoid activation was not
% delivering good performance for this set of parameters with training accuracy
% of 75% and testing accuracy of 30%
tic;
[wLayerHid,wLayerOut]=nnTanh(tfTrainPC', vLabelTraining', 25);
predClassSig=predictSig(tfTrainPC',wLayerHid,wLayerOut);
predClassSigT=predictSig(tfTestPC',wLayerHid,wLayerOut);
time_nnTanh=toc/60;
acc_nnSig=mean(predClassSig'==labelsTraining);
acc_nnSigT=mean(predClassSigT'==labelsTesting);
[acc_nnSig, acc_nnSigT]

% NEURAL NETWORK3: DEEP NEURAL NETWORK - 2 HIDDEN LAYERS-----------------
% To continue our experiment we test a NN with four layers (deep network): 
% inputs, two hidden layers and outputs. In the hidden layers we set ## neurons
% The returned improved results than the log and tanh model however still
% below our expectations - 66%.
tic;
[wLayerHid1,wLayerHid2,wLayerOut]=nndeep(tfTrainPC', vLabelTraining', 25, 15);
predClassDeep=predictdeep(tfTrainPC',wLayerHid1,wLayerHid2,wLayerOut);
predClassDeepT=predictdeep(tfTestPC',wLayerHid1,wLayerHid2,wLayerOut);
time_nnDeep=toc/60;
acc_nnDeep=mean(predClassDeep'==labelsTraining);
acc_nnDeepT=mean(predClassDeepT'==labelsTesting);
[acc_nnDeep, acc_nnDeepT]

% NEURAL NETWORK4: RELU ACTIVATION FUNCTION------------------------------
% During our research we found that RELU activation function is a very 
% common one, as is used in convolutional networks with great result.
% We decide to experiment with the RELU activation function. However the 
% results where not as expected, test set was unable to improve above 60%
% of accuracy changing different parameters of the model.
tic;
[wLayerHid,wLayerOut]=nnRelu(tfTrainPC', vLabelTraining', 25);
predClassRelu=predictRelu(tfTrainPC',wLayerHid,wLayerOut);
predClassReluT=predictRelu(tfTestPC',wLayerHid,wLayerOut);
time_nnRelu=toc/60;
acc_nnRelu=mean(predClassRelu'==labelsTraining);
acc_nnReluT=mean(predClassReluT'==labelsTesting);
[acc_nnRelu, acc_nnReluT]

% SUPPORT VECTOR MACHINE POLYNOMIAL KERNEL-------------------------------
% To end our research, we test SVMs. We decide to not test a linear kernel
% based on the results of the linear classifier, in this manner we decide
% to test a polynomial kernel initially. We test from 2 degrees to 9
% degrees of the polynomial. We end up with degree 3 having better results
% than the others with C = 2.
tic;
lambdaSVMP=svm(tfTrainPC', labelsTraining',2);
predClassSVM=predictsvm(lambdaSVMP,tfTrainPC',labelsTraining',tfTrainPC');
predClassSVMT=predictsvm(lambdaSVMP,tfTrainPC',labelsTraining',tfTestPC');
time_svmP=toc/60;
acc_svmP=mean(predClassSVM'==labelsTraining);
acc_svmPT=mean(predClassSVMT'==labelsTesting);
[acc_svmP, acc_svmPT]

% SUPPORT VECTOR MACHINE RADIAL BASIS FUNCION KERNEL---------------------
% Finally, we test with a RBF kernel since based on our research is the
% kernel with better results and more suitable for this type of problem. We
% test different combinations of C and gamma values finding the best of all
% C=2 and gamma=4 with an much higher results in our test data of 73%. We
% also test C=1 and 0.4 and gamma=5, 6 with the same results. Increasing
% the value of gamma above 6 reduce our accuracy and less than 4 
% reduce our accuracy as well.
tic;
lambdaRBF=svmrbf(tfTrainPC', labelsTraining',2,4); % TEST DIFFERENT SETTINGS
predClassRBF=predictsvmrbf(lambdaRBF,tfTrainPC',labelsTraining',tfTrainPC',4);
predClassRBFT=predictsvmrbf(lambdaRBF,tfTrainPC',labelsTraining',tfTestPC',4);
time_RBF=toc/60;
acc_RBF=mean(predClassRBF'==labelsTraining);
acc_RBFT=mean(predClassRBFT'==labelsTesting);
[acc_RBF, acc_RBFT]

%=============== CROSS VALIDATION SVM-RBF | K-NN MODEL ===================
tic;
kfold=10;
for k=1:kfold
	testind=randomIndexes(1:100);
	randomIndexes=randomIndexes(101:end);
	CVdataTrain=digitsData(randomIndexes,:);
	CVlabelsTrain=labels(randomIndexes);
	CVdataTest=digitsData(testind,:);
	CVlabelsTest=labels(testind);
	CVTrain=pcaTransform(CVdataTrain, explainedVar, CVdataTrain);
	CVTest=pcaTransform(CVdataTrain, explainedVar, CVdataTest);
	%SVMRBF model
	CVlmbdRBF=svmrbf(CVTrain', CVlabelsTrain',1,5);
	CVClassRBF=predictsvmrbf(CVlmbdRBF,CVTrain',CVlabelsTrain',CVTrain',5);
	CVClassRBFT=predictsvmrbf(CVlmbdRBF,CVTrain',CVlabelsTrain',CVTest',5);
	acc_RBF(k)=mean(CVClassRBF'==CVlabelsTrain);
	acc_RBFT(k)=mean(CVClassRBFT'==CVlabelsTest);
	%KNN model
	CVknn=knn(CVlabelsTrain',CVTrain',CVTrain',2);
	CVknnT=knn(CVlabelsTrain',CVTrain',CVTest',2);
	acc_knn(k)=mean(CVknn'==CVlabelsTrain);
	acc_knnT(k)=mean(CVknnT'==CVlabelsTest);
	
	randomIndexes=[randomIndexes,testind];
end
[acc_RBF; acc_RBFT]
 mean(acc_RBF), mean(acc_RBFT)
figure;
histogram(acc_RBFT,6)
title('Cross-validation kfold - 10 folds - RBF');

[acc_knn; acc_knnT]
 mean(acc_knn), mean(acc_knnT)
figure; histogram(acc_knnT,6);
title('Cross-validation kfold - 10 folds - K-NN');