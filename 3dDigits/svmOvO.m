function [lambda,indexes]=svmOvO(traindata, trainclass, C)
% using One-against-all approach
	cmbcount=1;
	for i=0:length(unique(trainclass))-1
		for j=i+1:length(unique(trainclass))-1
			clear K;
			indexes(:,cmbcount)=[i;j];
			x=traindata(:,trainclass==i | trainclass==j);
			y=trainclass(trainclass==i | trainclass==j);
			y(y==j)=-1;
			y(y==i)=1;
			myH=0;
			myf=0;
			myAeq=0;
			mybeq=0;
			myLB=0;
			myUB=0;	
		% DEFINE KERNEL (Xi'*Xj + constant)^degree (constant=1, degree=2)
			constant=1;
			degree=3;
			for ki=1:size(x,2)
				for kj=1:size(x,2)
					K(ki,kj)=polKern(x(:,ki),x(:,kj),constant,degree);
				end
			end
			myH=(y'*y).*K;
			myf=-ones(size(x,2),1);
			myAeq=y;
			mybeq=0;
			myLB=zeros(size(x,2),1);
			myUB=C*ones(size(x,2),1);
			myA=[];
			myb=[];
			lambda{:,cmbcount}=quadprog(myH,myf,myA,myb,myAeq,mybeq,myLB,myUB);
			cmbcount=cmbcount+1;
		end
	end
