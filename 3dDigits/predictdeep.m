function predictClass=predictdeep(testdata,wLayerHid1,wLayerHid2,wLayerOut)
% predictdeep function determines the class for the deep neural network.
a=1;
testdt=testdata;
testdt(end+1,:)=1;

vLayerHid1=wLayerHid1'*testdt; %update value input hidden layer with test data
yLayerHid1=1./(1+exp(-a.*vLayerHid1));%update output hidden layer Logistic function
yLayerHid1(end+1,:)=1; %update extended output hidden layer
vLayerHid2=wLayerHid2'*yLayerHid1; %update total net input second hidden layer
yLayerHid2=1./(1+exp(-a.*vLayerHid2));%update output second hidden layer
yLayerHid2(end+1,:)=1;%update extended output second hidden layer
vLayerOut=wLayerOut'*yLayerHid2; %total net input output layer
yLayerOut=1./(1+exp(-a.*vLayerOut));%output output layer
predClassDeep=yLayerOut==max(yLayerOut); %predict class
for i=1:size(predClassDeep,2)
	if max(predClassDeep(:,i)')>0 & sum(predClassDeep(:,i)')==1
		predictClass(i)=find(predClassDeep(:,i)'==1)-1;		
	else
		predictClass(i)=10;
	end
end