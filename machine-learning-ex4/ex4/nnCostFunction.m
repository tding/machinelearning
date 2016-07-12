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
X = [ones(m,1) X]; % add bias unit 
y_vector = zeros(m,num_labels); % convert label to vector 5000 * 10
for i=1:m
    y_vector(i,y(i)) = 1;
end;

% a1 = x
% a2 = sigmoid(X * Theta1') (5000 * 25)
a2 = sigmoid(X * Theta1');
a2 = [ones(m,1) a2]; % 5000 * 26

a3 = sigmoid(a2 * Theta2'); % 5000*10 hypothesis for each case
%Cost function
J = (1/m) * sum ( sum (  (-y_vector) .* log(a3)  -  (1-y_vector) .* log(1-a3) ));

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
sigma_3 = a3 - y_vector;
z2 = X * Theta1';
sigma_2 = (sigma_3*Theta2).* sigmoidGradient([ones(size(z2,1),1) z2]);
sigma_2 = sigma_2(:,2:end);

%accumulate gradients
delta_1 = sigma_2' * X;
delta_2 = sigma_3' * a2;

Theta1_grad = delta_1./m;
Theta2_grad = delta_2./m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Regularization
theta1_new = Theta1(:,2:size(Theta1,2)); % 25 * 400
theta2_new = Theta2(:,2:size(Theta2,2)); % 10 * 25

%Two layers
%First layers
layer1 = sum(sum(theta1_new .^2));
layer2 = sum(sum(theta2_new .^2));
reg = lambda / (2*m) * (layer1 + layer2);

J = J + reg;
grad_reg1 = (lambda / m) * [zeros(size(Theta1,1),1) theta1_new];
grad_reg2 = (lambda / m) * [zeros(size(Theta2,1),1) theta2_new];

Theta1_grad = Theta1_grad + grad_reg1;
Theta2_grad = Theta2_grad + grad_reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
