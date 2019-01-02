function [wLayerHid1,wLayerHid2,wLayerOut]=nndeep(traindata, trainclass, neurons1, neurons2)
% deep function is a deep neural network with two hidden layers. As inputs
% receives the number of neurons for each layer. As outputs returns the
% weight vector for each layer.

a=1; % Activation function: Logistic function (parameter a=1)
x=traindata;
x(end+1,:)=1; % extended data matrix
%initialize values for backpropagation algorithm
wLayerHid1=rand(size(x,1), neurons1)-0.5; %initialize hidden weights
vLayerHid1=wLayerHid1'*x; %total net input perceptron
yLayerHid1=1./(1+exp(-a.*vLayerHid1));%output hidden perceptron Logistic function
yLayerHid1(end+1,:)=1; %extended output hidden layer
wLayerHid2=rand(neurons1+1, neurons2)-0.5; %initialize second hidden weights
vLayerHid2=wLayerHid2'*yLayerHid1; %total net input second hidden layer
yLayerHid2=1./(1+exp(-a.*vLayerHid2));%output second hidden layer
yLayerHid2(end+1,:)=1; %extended output second hidden layer
wLayerOut=rand(neurons2+1, size(trainclass,1))-0.5; %initialize output weights
vLayerOut=wLayerOut'*yLayerHid2; %total net input output layer
yLayerOut=1./(1+exp(-a.*vLayerOut));%output output layer

lr=0.01; %learning rate for the backpropagation algorithm
% After initializing all the algorithm variables, we can start the
% iterations
iteration=1;
while iteration<=100000
	% determine cost function
	errorTotal(iteration)=sum(sum(1/2.*(trainclass-yLayerOut).^2)); %calculate error
	if errorTotal(iteration)<1e-4 % the learning is good enough
		break;
	end

	%first we calculate the delta of the output layer
	deltawLayerOut=...
		-(trainclass-yLayerOut).*... % output error
		yLayerOut.*(ones(size(yLayerOut))-yLayerOut)*... % activation function derivate
		yLayerHid2'...
		; 
	%then we calculate the delta of the hidden2 layer
	deltawLayerHid2=...
		wLayerOut*...
		(-(trainclass-yLayerOut).*... % output error
		yLayerOut.*(ones(size(yLayerOut))-yLayerOut)).*... % activation function derivate
		yLayerHid2.*(ones(size(yLayerHid2))-yLayerHid2)*... % activation function derivate hidden layer
		yLayerHid1'...
		;
	%then we calculate the delta of the hidden1 layer
	deltawLayerHid1=...
		wLayerHid2*...
		(wLayerOut(1:end-1,:)*...
		(-(trainclass-yLayerOut).*... % output error
		yLayerOut.*(ones(size(yLayerOut))-yLayerOut)).*... % activation function derivate
		yLayerHid2(1:end-1,:).*(ones(size(yLayerHid2(1:end-1,:)))-yLayerHid2(1:end-1,:))).*... % activation function derivate
		yLayerHid1.*(ones(size(yLayerHid1))-yLayerHid1)*... % activation function derivate
		x'...
		;
	%update the output and hidden weights
	wLayerHid1=wLayerHid1-lr*deltawLayerHid1(1:end-1,:)'; % update old weights to new ones
	wLayerHid2=wLayerHid2-lr*deltawLayerHid2(1:end-1,:)'; % update old weights to new ones
	wLayerOut=wLayerOut-lr*deltawLayerOut'; % update last layer old weights to new ones

	%update values with new weights
	vLayerHid1=wLayerHid1'*x; %update value input hidden layer
	yLayerHid1=1./(1+exp(-a.*vLayerHid1));%update output hidden perceptron Logistic function
	yLayerHid1(end+1,:)=1; %update extended output hidden layer
	vLayerHid2=wLayerHid2'*yLayerHid1; %update total net input second hidden layer
	yLayerHid2=1./(1+exp(-a.*vLayerHid2));%update output second hidden layer
	yLayerHid2(end+1,:)=1;%update extended output second hidden layer
	vLayerOut=wLayerOut'*yLayerHid2; %total net input output layer
	yLayerOut=1./(1+exp(-a.*vLayerOut));%output output layer
	iteration=iteration+1;
end