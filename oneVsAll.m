function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

m = size(X, 1); % # of training examples
n = size(X, 2); % # of features
initial_theta = zeros(n+1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Add ones to the X data matrix
X = [ones(m, 1), X];

% Each row contains parameters for respective label.
all_theta = zeros(num_labels, n+1);

% Compute optimal parameters for each label.
for current_label=1:num_labels
	all_theta(current_label,:) = fmincg(
		@(t)(lrCostFunction(t, X, (y == current_label), lambda)),
		initial_theta, 
		options
	);
end

end