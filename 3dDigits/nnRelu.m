function [wLayerHid,wLayerOut]=nnRelu(traindata, trainclass, neurons)

x=traindata;
a=0.01;
x(end+1,:)=1; % extended data matrix
%initialize values for backpropagation algorithm
wLayerHid=rand(size(x,1), neurons)-0.5; %initialize hidden weights
vLayerHid=wLayerHid'*x; %total net input perceptron
yLayerHid=max(a*vLayerHid,vLayerHid);%output hidden perceptron Logistic function
yLayerHid(end+1,:)=1; %extended output hidden layer
wLayerOut=rand(neurons+1, size(trainclass,1))-0.5; %initialize output weights
vLayerOut=wLayerOut'*yLayerHid; %total net input output layer

%yLayerOut=max(a*vLayerOut,vLayerOut);%output output layer
yLayerOut=1./(1+exp(-a.*vLayerOut));

lr=0.01; %learning rate for the backpropagation algorithm
% After initializing all the algorithm variables, we can start the
% iterations
iteration=1;
while iteration<100000
	% determine cost function
	errorTotal(iteration)=sum(sum(1/2.*(trainclass-yLayerOut).^2)); %calculate error
	if errorTotal(iteration)<1e-4 % the learning is good enough
		break;
	end
	%update weights

	%deltaw=-(trainclass-outputNN)*outputNN*(1-outputNN)*outputperceptron
	%first we calculate the delta of the output layer
	
	%derivLayerOut=yLayerOut./vLayerOut;
	derivLayerOut=yLayerOut.*(ones(size(yLayerOut))-yLayerOut);
	deltawLayerOut=...
		-(trainclass-yLayerOut).*... % output error
		derivLayerOut*... % activation function derivate
		yLayerHid'...
		; 
	%then we calculate the delta of the hidden layer
	derivLayerHid=yLayerHid(1:end-1,:)./vLayerHid;
	derivLayerHid(end+1,:)=1;
	deltawLayerHid=...
		wLayerOut*...
		(-(trainclass-yLayerOut).*... % output error
		derivLayerOut).*... % activation function derivate
		derivLayerHid*... % activation function derivate
		x'...
		;
	%update the output and hidden weights
	wLayerHid=wLayerHid-lr*deltawLayerHid(1:end-1,:)'; % update w11 old weights to new ones
	wLayerOut=wLayerOut-lr*deltawLayerOut'; % update last layer old weights to new ones

	%update values with new weights
	vLayerHid=wLayerHid'*x; %update value input hidden layer
	yLayerHid=max(a*vLayerHid,vLayerHid);%update output hidden perceptron Logistic function
	yLayerHid(end+1,:)=1; %update extended output hidden layer
	vLayerOut=wLayerOut'*yLayerHid; %update total net input output layer
	%yLayerOut=max(a*vLayerOut,vLayerOut);%update output output layer
	yLayerOut=1./(1+exp(-a.*vLayerOut));
	iteration=iteration+1;
end