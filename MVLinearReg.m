clear ; close all; clc;

X(:,1) = rand(100,1);
X(:,2) = rand(100,1);
y = 10 + 2*X(:,1) - 3*X(:,2); 
m = length(y);



X = [ones(m,1) X];

%Intialize theta to be a n+1 dimension vector with all elements equal to zero
theta = zeros(size(X,2),1);

costFunction(X, y, theta);

%Setting variables for gradient descent
alpha = 0.9;
num_iter = 200;


[theta, J_history]  = gradientDescent(X,y,alpha,num_iter,theta);

%Checking the convergence
figure;
plot(1:numel(J_history), J_history, 'r', 'LineWidth', 2);
xlabel('Number of Iterations');
ylabel('Value of J(theta)');


disp(sprintf('theta value:'));
fprintf('%f\n', theta);

y_predicted = X * theta;

rmse = sqrt( (1/m) * ((y_predicted - y) .^ 2) );
rmse = sum(rmse);
disp(sprintf('RMSE:'));
fprintf('%f\n', rmse);
