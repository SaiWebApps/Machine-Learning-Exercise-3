function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); % # of training examples
num_labels = size(all_theta, 1); % # of output classes (multi-class)
X = [ones(m, 1), X]; % Add column of ones to the X data matrix (x0).

% all_theta = result of oneVsAll (num_labels x (n+1))
% X = (m x (n+1))
% h(i,j) = probability that example i is label j
h = X * all_theta'; % (m x num_labels)

% p -> (mx1) vector, where p(i) = prediction for training example i
% We make our prediction by selecting the label with the highest probability
% for each training example in h.
[maxProbabilities, p] = max(h, [], 2); % Row-wise maxima

end