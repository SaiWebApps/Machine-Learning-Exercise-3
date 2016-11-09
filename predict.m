function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1); % # of training examples
num_labels = size(Theta2, 1); % # of output classes
X = [ones(m,1), X]; % Add column of 1s for X0 to input X.

hiddenLayer = sigmoid(X * Theta1'); % (5000x401) * (401x25) = (5000x25)
% Add activation unit 0 (+1) for each training example.
hiddenLayer = [ones(m, 1), hiddenLayer]; % (5000x26)

% Each element (i,j) represents the probability that
% training example i is label j.
outputLayer = sigmoid(hiddenLayer * Theta2'); % (5000x26) * (26x10) = (5000x10)

% Now, find the label corresponding to the maximum probability for
% each training example; this is our prediction for each example.
[maxProbabilities, p] = max(outputLayer, [], 2); % p -> 5000x1

end