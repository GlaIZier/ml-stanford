function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

diff = (X * Theta' - Y); % nm x nf (number of features) X nf x nu = nm x nu
squared_diff = diff .^ 2;
J = (1 / 2) * sum(squared_diff(R == 1)); % calculate sum only where R is 1
J_reg = (lambda / 2) * (sum(Theta(:) .^ 2) + sum(X(:) .^ 2));
J = J + J_reg;

% iterative approach
%for i = 1: num_movies
%  for j = 1 : num_users
%    if (R(i, j) == 1)
%      for k = 1: num_features
%        d = Theta(j, :) * X(i, :)' - Y(i, j); % 1  
%        sX_grad = d * Theta(j, k);
%        sTheta_grad = d * X(i, k);
%        X_grad(i, k) = X_grad(i, k) + sX_grad;
%        Theta_grad(j, k) = Theta_grad(j, k) + sTheta_grad;
%      end
%    end
%  end
%end

% more vectorized approach
%for i = 1: num_movies
%  for j = 1 : num_users
%    if (R(i, j) == 1)
%      d = Theta(j, :) * X(i, :)' - Y(i, j); % 1
%      X_grad(i, :) = X_grad(i, :) + d * Theta(j, :); % 1 * 1 x nf
%      Theta_grad(j, :) = Theta_grad(j, :) + d * X(i, :);
%    end
%  end
%end

% solution from the ex8.pdf
for i = 1: num_movies
  idx = find(R(i,:) == 1); % 1 x r(nu that voted)
  Theta_temp = Theta(idx, :); % r x nf
  Y_temp = Y(i, idx); % 1 x r
  d_vectorized = X(i, :) * Theta_temp' - Y_temp; %  1 x nf X nf x r - 1 x r 
  X_grad(i,:) = d_vectorized * Theta_temp; % 1 x r X r x nf (this is where summation is happening for a user r x r) = 1 x nf
  X_grad_reg = lambda * X(i, :);
  X_grad(i,:) = X_grad(i,:) + X_grad_reg;
end

for j = 1: num_users
  idx = find(R(:,j) == 1); % r(nm that were rated by j user) x 1
  Y_temp = Y(idx, j); % r x 1
  X_temp = X(idx, :); % r x nf
  d_vectorized = X_temp * (Theta(j,:))' - Y_temp; % r x nf X nf x 1 - r x 1
  Theta_grad(j,:) =  d_vectorized' * X_temp; % 1 x r X r x nf (r * r summation of error for a movie)
  Theta_grad_reg = lambda * Theta(j, :);
  Theta_grad(j,:) = Theta_grad(j,:) + Theta_grad_reg;
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
