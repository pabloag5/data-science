function tfPC=pcaTransform(dataTraining, explainedVar, dataTesting)
% dataTraining is used to determine the eigenvectors
% explainedVar is the porcentage of variance we want to be explained by the
% eigenvectors
% dataTesting is the data we want to transform to the eigenvectors space
% tfPC is the result data by the pca transformation with dimensionality
% according to the desire explained variance.
dtTrainCov=cov(dataTraining);
[eigVecTrain, eigValTrain]=eig(dtTrainCov);
totalEigVal=sum(diag(eigValTrain));
varPropEigVec=diag(eigValTrain)/totalEigVal;
[varPropEigVecD,sortInd]=sort(varPropEigVec,'descend');
eigValTraind=eigValTrain(:,sortInd);
eigVecTraind=eigVecTrain(:,sortInd);
tfFeatures=sum(cumsum(varPropEigVecD)<=explainedVar);
% Create the transformed data matrix in the principal components space defined.
tfdata=dataTesting*eigVecTraind;
tfPC=tfdata(:,1:tfFeatures);