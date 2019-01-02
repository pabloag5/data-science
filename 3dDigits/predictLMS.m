function predictClass=predictLMS(data,w,wind)
% predictClass predicts the outcome of the LMS algorithm.
% The inputs are the data to be test, the weights matrix where each column represents the weights of
% the corresponding combination of classes according to wind matrix; and
% the weight indexes matrix which contains the indexes of the classes for each train model
% where first row are to positive discriminants and second row are the
% negative discriminants for each model trained.
% The process determines the class for each run and define the final class as
% the class most often for each sample.
	clear predictClasstmp predictClass
	test=data;
	test(end+1,:)=1;
	for wcount=1:size(wind,2)
		wtest=w(:,wcount)'*test;
		predictClasstmp(wcount,:)=wtest;
		%predictClass=zeros(1,size(wtest,2));
		predictClasstmp(wcount,wtest>0)=wind(1,wcount);
		predictClasstmp(wcount,wtest<0)=wind(2,wcount);
	end
	predictClass=mode(predictClasstmp);