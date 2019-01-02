% script to import the dataset
function data=loading_data
locationData='digits_3d_training_data\digits_3d\training_data\';
dataFiles=dir(fullfile(locationData,'*.mat'));
data=[];
for i = 1:length(dataFiles)
	fullFileName=fullfile(locationData, dataFiles(i).name);
	tmp=importdata(fullFileName);
    data = [data; tmp i*ones(size(tmp,1),1)];
end
