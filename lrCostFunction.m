function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
n = length(theta); % number of features
theta1n = theta(2:n); % theta(1) = theta0; this is theta1 to thetaN

% Compute unregularized cost function.
h = sigmoid(X * theta);
J0 = y' * log(h); % Cost for "y=0" class predictions
J1 = (1 - y)' * log(1 - h); % Cost for "y=1" class predictions
JUnregularized = (-1/m) * (J0 + J1);

% Compute regularized cost function.
sumTheta1nSquared = theta1n' * theta1n; % Sum of squares of theta1 to thetaN
regularizationCost = (lambda / (2*m)) * sumTheta1nSquared;
J = JUnregularized + regularizationCost;

% Compute gradients (partial derivatives of cost function).
grad = (1/m) * (X' * (h-y)); % Unregularized
grad(2:n) = grad(2:n) + ((lambda / m) * theta1n); % Regularized

end