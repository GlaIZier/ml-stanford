function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -- Part 1
% Feedforward

X = [ones(m, 1) X]; % m x 401
a1 = X;

z2 = Theta1 * a1'; % 25 x 401 X 401 x m
a2 = sigmoid(z2); % 25 x m 
a2 = a2'; % m x 25

a2 = [ones(m, 1) a2]; % m x 26
z3 = Theta2 * a2'; % num_labels(= 10) x 26 X 26 x m = num_labels x m
h = sigmoid(z3); % num_labels x m
h = h'; % m x num_labels
a3 = h;

% y -> zero-one matrix (the same as h)
y_matrix = zeros(size(y, 1), num_labels); % m x num_labels(= 10)
for i = 1 : num_labels
  y_matrix(:, i) = (y == i);
end

% J 
minuendLog = log(h); % m x num_labels
minuend = y_matrix .* minuendLog; % m x num_labels
subtrahendLog = log(1 - h);
subtrahend = (1 - y_matrix) .* subtrahendLog;
diff = -minuend - subtrahend; % m x 10
% We vectorize diff to (m * num_labels x 1) because octave can't calculate sum(diff, 'all') as in Matlab.
% Sum here is actually a double sum since we our matrix is m x num_labels. 
% m is i (rows) and num_labels is K (columns) in the cost formula
J = (1 / m) * sum(diff(:)); 

% Regularization
Theta1_unbiasied = Theta1(:, 2 : size(Theta1, 2)); 
Theta2_unbiasied = Theta2(:, 2 : size(Theta2, 2));
Theta1_unbiasied_sqr = Theta1_unbiasied .^ 2;
Theta2_unbiasied_sqr = Theta2_unbiasied .^ 2;

regularization = (lambda / (2 * m)) * (sum(Theta1_unbiasied_sqr(:)) + sum(Theta2_unbiasied_sqr(:))); % vectorize Thetas to sum for all dims

J = J + regularization;



% -- Part 2
% Backpropagation

D1 = zeros(size(Theta1_grad));
D2 = zeros(size(Theta2_grad));
for t = 1: m 
  % 1 feedprop. Already done
  
  % 2
  a3t = (a3(t, :))'; % num_labels x 1
  yt = (y_matrix(t, :))'; % num_labels x 1
  delta3 = a3t - yt; % num_labels x 1
  
  % 3
  a2t = (a2(t, :))'; % m x 26 -> 26 x 1
  sg = a2t .* (1 - a2t);   % 26 x 1
  multiplier = Theta2' * delta3; % 26 x num_labels X num_labels x 1 = 26 x 1
  delta2 = multiplier .* sg; % 26 x 1
  delta2 = delta2(2 : end); % 25 x 1. Exclude biased delta
  
  % 4
  a1t = (a1(t, :))'; % m x 401 -> 401 x 1
  D2 = D2 + delta3 * a2t'; % num_labels x 1 X 1 x 26 = num_labels x 26
  D1 = D1 + delta2 * a1t'; % 25 x 1 X 1 x 401 = 25 x 401
  
end
% 5
Theta1_grad = (1 / m) * D1;
Theta2_grad = (1 / m) * D2;

% gradient regularization
Theta1_grad_bias = Theta1_grad(:, 1); % save first (unbiased) column (j = 0)
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta1_grad(:, 1) = Theta1_grad_bias;

Theta2_grad_bias = Theta2_grad(:, 1); % save first (unbiased) column (j = 0)
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
Theta2_grad(:, 1) = Theta2_grad_bias;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
