function [error_test] = testError(X, y, Xtest, ytest)

lambda  = 3;
theta = trainLinearReg(X,y,lambda);
[Jtest,grad_train] = linearRegCostFunction(Xtest, ytest, theta, 0);

error_test = Jtest;

% =========================================================================

end