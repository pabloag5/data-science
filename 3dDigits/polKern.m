function krnl = polKern(x1, x2,constant,degree)
% this function calculates the polynomial kernel based on the constant and
% degree paramaters as inputs.

krnl = (x1' * x2 + constant).^degree;