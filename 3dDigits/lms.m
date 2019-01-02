function [w,wind] = lms(class, data)
% Function to calculate weights for each feature. The method OneVSOne, e.g:
% w01 -> class=0 vs class=1, w67 -> class=6 vs class 7.
% The inputs are the train class as 0,1,2,..9 and the train dataset.
% The outputs are a w matrix where each column represents the weights of
% the corresponding combination of classes according to wind matrix.
% wind matrix contains the indexes of the classes for each train model
% where first row are to positive discriminants and second row are the
% negative discriminants for each model trained.

wcount=1;
	for i=0:length(unique(class))-1
		for j=i+1:length(unique(class))-1
			wind(:,wcount)=[i;j];
			x=data(:,class==i | class==j);
			y=class(class==i | class==j);
			x(end+1,:)=1;
			y(y==j)=-1;
			y(y==i)=1;
			%initializing w and wx
			w(:,wcount)=(rand(size(x,1),1) - 0.5) / 10;
			w(:,wcount)=w(:,wcount)./norm(w(:,wcount));
			xw=x'*w(:,wcount);
			xw(xw>0)=1;
			xw(xw<0)=-1;
			ro=0.01;
			process=1;
			iter=20000;
			wold=ones(size(x,1), 1);

			%batch perceptron algorithm
			while ...%norm(w(:,wcount)./norm(w(:,wcount))-wold./norm(wold))>1e-9 && ...
					process<iter && ...
					sum(x*(xw-y')==0)~=size(data,2)
				
				wold=w(:,wcount);
				dJw=x*(y'-xw);
				w(:,wcount)=w(:,wcount)+(ro*dJw);
				w(:,wcount)=w(:,wcount)./norm(w(:,wcount));
				xw=x'*w(:,wcount);
				xw(xw>0)=1;
				xw(xw<0)=-1;
				process=process+1; 
				ro=ro/process; %test if is better
			end
		wcount=wcount+1;
		end
	end