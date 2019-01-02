function lambda=svmrbf(traindata, trainclass,C,gamma)
% svmrbf function calculates the lambdas for each class. The lambdas are
% used to calculate the support vectors and the discriminant functions for
% each sample and each class. It uses the radial basis function kernel.
% It uses One-against-all approach for the multiclass analysis
y=trainclass;
x=traindata;
% C=2;
% gamma=4;
	for class=0:9
		y=trainclass;
		y(y~=class)=-1;
		y(y==class)=1;
		myH=0;
		myf=0;
		myAeq=0;
		mybeq=0;
		myLB=0;
		myUB=0;	
	% DEFINE KERNEL 
		for i=1:size(x,2)
			for j=1:size(x,2)
				K(i,j)=rbfKern(x(:,i),x(:,j),gamma);
			end
		end
		myH = (y' * y).*K;
		myf=-ones(size(traindata,2),1);
		myAeq=y;
		mybeq=0;
		myLB=zeros(size(traindata,2),1);
		myUB=C*ones(size(traindata,2),1);
		myA=[];
		myb=[];
		lambda(:,class+1)=quadprog(myH,myf,myA,myb,myAeq,mybeq,myLB,myUB);
	end