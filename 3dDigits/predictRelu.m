function predictClass=predictRelu(testdata,wLayerHid,wLayerOut)
% predictRelu function predict the class with the neural network with
% RELU activation function as inputs.
a=0.01;
testdt=testdata;
testdt(end+1,:)=1;
vLayerHid=wLayerHid'*testdt;
yLayerHid=max(a*vLayerHid,vLayerHid);
yLayerHid(end+1,:)=1;
vLayerOut=wLayerOut'*yLayerHid;
%yLayerOut=max(a*vLayerOut,vLayerOut);
yLayerOut=1./(1+exp(-a.*vLayerOut));
predClassRelu=yLayerOut==max(yLayerOut);
for i=1:size(predClassRelu,2)
	if max(predClassRelu(:,i)')>0 & sum(predClassRelu(:,i)')==1
		predictClass(i)=find(predClassRelu(:,i)'==1)-1;		
	else
		predictClass(i)=10;
	end
end