function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%sum_errors  = (1/(2*m)) * (sum( ((X * theta) - y) .^ 2));
% Notice how we DO NOT regularize the first parameter
%reg_term  = (lambda/(2*m)) * sum(theta(2:end) .^ 2);
% And finally our cost function
%J = sum_errors + reg_term;

 h =(X*theta);
 temp=(h-y);
 as1 = (1/(2*m))*(temp'*temp);
 as2 = (lambda/(2*m))*(theta(2:end)'*theta(2:end));
 J=as1+as2;
 grad = (1/m)*(X'*((X * theta)-y));
 grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);









% =========================================================================

grad = grad(:);

end
