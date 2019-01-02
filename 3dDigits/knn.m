function C=knn(trainclass,traindata,data,k)
% K-NN: this function calculates the euclidien distance between the
% testdata points to the traindata points and determine the class based on
% the number of clusters define as inputs

	C=[];
	for i=1:size(data,2)
		%create euclidien distance matrix between sample and traindata
		dataDist=dist([data(:,i),traindata]);
		%sort sample distances to traindata
		[valData indData]=sort(dataDist(1,2:end),'ascend');
		%find k nearest neighbours
		dataKNN=trainclass(indData(1:k));
		%group nearest classes
		[grp val]=findgroups(dataKNN);
		nn=[val; splitapply(@length,dataKNN,grp)];
		%find the knn class
		kadjust=1; %initialize value to adjust k
		while size(nn(1,find(nn(2,:)==max(nn(2,:)))),2) > 1
			dataKNN=trainclass(indData(1:k-kadjust));
			%group nearest classes
			[grp val]=findgroups(dataKNN);
			nn=[val; splitapply(@length,dataKNN,grp)];
			kadjust=kadjust+1;
		end
		%store class to vector
		C(i)=nn(1,find(max(nn(2,:))));
	end
end