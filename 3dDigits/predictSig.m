function predictClass=predictSig(testdata,wLayerHid,wLayerOut)
% predictSig function predict the class with the neural network with
% sigmoid activation function (tanh) as inputs
testdt=testdata;
testdt(end+1,:)=1;
vLayerHid=wLayerHid'*testdt;
yLayerHid=tanh(vLayerHid);
yLayerHid(end+1,:)=1;
vLayerOut=wLayerOut'*yLayerHid;
yLayerOut=tanh(vLayerOut);
predClassSig=yLayerOut==max(yLayerOut);
for i=1:size(predClassSig,2)
	if max(predClassSig(:,i)')>0 & sum(predClassSig(:,i)')==1
		predictClass(i)=find(predClassSig(:,i)'==1)-1;		
	else
		predictClass(i)=10;
	end
end