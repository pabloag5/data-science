% preprocessing_data function imports the whole dataset using the 
% loading_data function, then starts a process which consists of the following:
%{
-Plotting the raw data to understand how the data was collected and recorded
-Initial phase of dimensionality reduction by only taking into account two
dimensions and the integer values of the coordinates.
-Second phase consists of taking unique values of the previous subprocess
-Third phase creates group levels of the longitute and latitute values
following by assigning each sample to respective groups. By this means we
reduce farther the number of dimensions and mapped the data to a grid type
of information.
-Finally, the last phase consists in vectorizing the data and create a
matrix where each row represents a sample and each column represent the
longitute and latitute coordinates in the grid, ending with a N x blocks^2
matrix size.
%}
function digitsVectors=preprocessing_data
close all;
data=loading_data;
% plotting some values to understand the data
figure;
indexes=randperm(100,10);
for i=1:10
	subplot(1,10,i)
	randDigit=find(data(:,4)==100*(i-1)+indexes(i));
	scatter(data(randDigit,1),data(randDigit,2));
end

% third variable is depthness of the movement.
% For this analysis we are not going to use this feature.
% In order to reduce dimensionality initially we are going to take into account
% the integer part of the features.
digits=round(data(:,[1 2 4]));
% As integers, some of the rows repeat itself, we are going to keep the
%unique values
cleanDigits=unique(digits,'rows');

%Understanding the data as a time series, if there is not enough movement
%by the human hand, the sensor records multiple coordinates in a tight
%space. To handle this we are going to set levels of the latitude and
%longitute scales and assign each value to the respective level. 
%Determine the levels of each feature: 
numlevels=20;
lonlevels=linspace(min(cleanDigits(:,1)),max(cleanDigits(:,1))+1,numlevels);
latlevels=linspace(min(cleanDigits(:,2)),max(cleanDigits(:,2))+1,numlevels);

%Assign each sample value to the respective level
longrp=discretize(cleanDigits(:,1),lonlevels);
latgrp=discretize(cleanDigits(:,2),latlevels);
%Now with the longitute and latitude groups we create a new dataset only
%taking into account unique values. By this mean we reduce the
%dimensionality and normalize the data.
gridDigits=unique([longrp,latgrp,cleanDigits(:,3)],'rows');

%Plotting to view the results of the preprocessing initial phase using the
%same digits for all subprocesses
figure;
for i=1:10
	% plotting raw data
	subplot(4,10,i)
	rawValues=find(data(:,4)==100*(i-1)+indexes(i));
	scatter(data(rawValues,1),data(rawValues,2));
	% plotting rounded values
	subplot(4,10,i+10)
	digitsValues=find(digits(:,3)==100*(i-1)+indexes(i));
	scatter(digits(digitsValues,1),digits(digitsValues,2));
	% plotting unique rounded values
	subplot(4,10,i+20)
	cleanValues=find(cleanDigits(:,3)==100*(i-1)+indexes(i));
	scatter(cleanDigits(cleanValues,1),cleanDigits(cleanValues,2));
	% plotting grouped values
	subplot(4,10,i+30)
	gridValues=find(gridDigits(:,3)==100*(i-1)+indexes(i));
	scatter(gridDigits(gridValues,1),gridDigits(gridValues,2));
end

% Create vectorize dataset: we create a matrix where each row represents a
% sample, since our grid is 20x20, each sample will have 20^2 values; 
% first 20 indexes represent values with longitute 1 and latitute between 1
% to 20, next 20 indexes (21 to 40) represent values with longitute 2 and
% latitute between 1 to 20 and so on until we get to longitute 20 (381-400) and
% latitute between 1 to 20; e.g:
% point (1,1) will be locate it index 1, (2,3) will be locate it at index
% 23.
digitsVectors=zeros(1000,numlevels^2);
lonindex=1:numlevels:numlevels^2;
for i=1:1000
	tmpDigit=gridDigits(gridDigits(:,3)==i,1:2);
	digitsVectors(i,lonindex(tmpDigit(:,1))+tmpDigit(:,2)'-1)=1;
end
% This ends the preprocessing phase of our data. Next we will split the
% vectorize dataset between train and test subsets.