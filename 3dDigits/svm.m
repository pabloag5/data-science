function lambda=svm(traindata, trainclass,C)
% svm function calculates the lambdas using a polynomial kernel of
% degree 2 and constant 1. The lambdas are
% used to calculate the support vectors and the discriminant functions for
% each sample and each class.
% It uses One-against-all approach for the multiclass analysis
	y=trainclass;
	x=traindata;
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
	% DEFINE KERNEL (Xi'*Xj + constant)^degree (constant=1, degree=2)
		constant=1;
		degree=3;
		for i=1:size(x,2)
			for j=1:size(x,2)
				K(i,j)=polKern(x(:,i),x(:,j),constant,degree);
			end
		end	
		myH=(y'*y).*K;
		myf=-ones(size(traindata,2),1);
		myAeq=y;
		mybeq=0;
		myLB=zeros(size(traindata,2),1);
		myUB=C*ones(size(traindata,2),1);
		myA=[];
		myb=[];
		lambda(:,class+1)=quadprog(myH,myf,myA,myb,myAeq,mybeq,myLB,myUB);
	end