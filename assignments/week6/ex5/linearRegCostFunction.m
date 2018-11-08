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

% J
h = X * theta; % m x 2 (n + 1) X 2 (n + 1) x 1 = m x 1
diff = h - y; % m x 1
J = (1 / (2 * m)) * sum(diff.^2);

penaltySum = sum(theta .^ 2) - theta(1) ^ 2; % excluding theta(1)
J = J + (lambda / (2 * m)) * penaltySum;

% grad

diffGrad = h - y; % m x 1
product = diffGrad .* X; % m x 2
total = sum(product, 1); % 1 x 2
total = total'; % 2 x 1
grad = (1 / m) * total;

gradFirst = grad(1); %save excluding element
grad = grad + (lambda / m) * theta;
grad(1) = gradFirst;

% =========================================================================

grad = grad(:);

end
