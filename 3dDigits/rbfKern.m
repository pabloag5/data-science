function krnl = rbfKern(x1, x2,gamma)
% this function calculates the radial basis function kernel
krnl = exp(-gamma*(dist(x1',x2)^2));