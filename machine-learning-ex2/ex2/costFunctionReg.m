function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
% j from 1 to n (theta[28,1])
temp = size(theta);
regvector = ones(1, temp(1));
regvector(1,1) = 0;
p = lambda/(2*m)* (regvector * (theta .^ 2));
J = (1/m) * (-y' * log(h)-(1-y)'*log(1-h)) + p;

regidimat = eye(temp(1));
regidimat(1,1) = 0;

% lambda * theta[28,1]' => [1,28] * [28,28] => [1,28]
gradreg = lambda * theta' * regidimat;

% [100,1] => [1,100] * [100,1] => [1] + [1,28]
grad = (1/m) *((h - y)' * X) + gradreg; 


% =============================================================

end
