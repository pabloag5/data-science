function predictClass=predictsvmrbf(lambda,traindata,trainclass,testdata,gamma)
% predictsvmrbf function receives as an input the lambda matrix from the rbf svm to calculate
% the support vector and discriminant function for each class. Finally determines the class based on the higher
% discriminant function value.	
	x=traindata;
	y=trainclass;
	for class=0:9
		y=trainclass;
		y(y~=class)=-1;
		y(y==class)=1;
	% support vectors
		svl=find(lambda(:,class+1)>1e-10);
		SV=x(:,svl);
		for i=1:size(SV,2)
			for j=1:size(testdata,2)
				Ktest(i,j)=rbfKern(SV(:,i),testdata(:,j),gamma);
			end
		end
		for i=1:size(testdata,2)
			ttclass(class+1,i)=...
				sign(sum(lambda(svl,class+1).*y(svl)'.*Ktest(:,i)));
			discfunction(class+1,i)=...
				sum(lambda(svl,class+1).*y(svl)'.*Ktest(:,i));
		end
		clear Ktest;
	end	
	for j=1:size(testdata,2)
		predictClass(j)=find(discfunction(:,j)==max(discfunction(:,j)))-1;
	end