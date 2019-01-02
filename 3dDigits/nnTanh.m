function [wLayerHid,wLayerOut]=nnTanh(traindata, trainclass, neurons)

a=1; % Activation function: Logistic function (parameter a=1)
x=traindata;
x(end+1,:)=1; % extended data matrix
%initialize values for backpropagation algorithm
wLayerHid=rand(size(x,1), neurons); %initialize hidden weights
vLayerHid=wLayerHid'*x; %total net input perceptron
yLayerHid=tanh(vLayerHid);%output hidden perceptron Logistic function
yLayerHid(end+1,:)=1; %extended output hidden layer
wLayerOut=rand(neurons+1, size(trainclass,1)); %initialize output weights
vLayerOut=wLayerOut'*yLayerHid; %total net input output layer
yLayerOut=tanh(vLayerOut);%output output layer
lr=0.005; %learning rate for the backpropagation algorithm
% After initializing all the algorithm variables, we can start the
% iterations
iteration=1;
while iteration<=100000
	% determine cost function
	errorTotal(iteration)=sum(sum(1/2.*(trainclass-yLayerOut).^2)); %calculate error
	if errorTotal(iteration)<1e-4 % the learning is good enough
		break;
	end
	%update weights

	%deltaw=-(trainclass-outputNN)*outputNN*(1-outputNN)*outputperceptron
	%first we calculate the delta of the output layer
	deltawLayerOut=...
		-(trainclass-yLayerOut).*... % output error
		 (1-yLayerOut.^2)*... % activation function derivate
		yLayerHid'...
		;
	%then we calculate the delta of the hidden layer
	deltawLayerHid=...
		wLayerOut*...
		-((trainclass-yLayerOut).*... % output error
		(1-yLayerOut.^2)).*... % activation function derivate
		(1-yLayerHid.^2)*... % activation function derivate
		x'...
		;
	%update the output and hidden weights
	wLayerHid=wLayerHid-lr*deltawLayerHid(1:end-1,:)'; % update w11 old weights to new ones
	wLayerOut=wLayerOut-lr*deltawLayerOut'; % update last layer old weights to new ones

	%update values with new weights
	vLayerHid=wLayerHid'*x; %update value input hidden layer
	yLayerHid=tanh(vLayerHid);%update output hidden perceptron Logistic function
	yLayerHid(end+1,:)=1; %update extended output hidden layer
	vLayerOut=wLayerOut'*yLayerHid; %update total net input output layer
	yLayerOut=tanh(vLayerOut);%update output output layer
	
	iteration=iteration+1;
end