function predictClass=predictLog(testdata,wLayerHid,wLayerOut)
% predictLog function predict the class with the neural network with
% logictic activation function as inputs
a=1;
testdt=testdata;
testdt(end+1,:)=1;
vLayerHid=wLayerHid'*testdt;
yLayerHid=1./(1+exp(-a.*vLayerHid));
yLayerHid(end+1,:)=1;
vLayerOut=wLayerOut'*yLayerHid;
yLayerOut=1./(1+exp(-a.*vLayerOut));
predClassLog=yLayerOut==max(yLayerOut);
for i=1:size(predClassLog,2)
	if max(predClassLog(:,i)')>0 & sum(predClassLog(:,i)')==1
		predictClass(i)=find(predClassLog(:,i)'==1)-1;		
	else
		predictClass(i)=10;
	end
end