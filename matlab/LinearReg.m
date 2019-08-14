clear ; close all; clc

%Reading the data
data = csvread('slr06_1.csv');

%Removing the header
data = data(2:size(data,1),:);


x = data(:,1);
y = data(:,2);
m = length(y);

figure;
plot(x, y, 'rx', 'MarkerSize', 15);
xlabel('Number of claims');
ylabel('Total payment for the claims in thousands of Swedish Konor');

X = [ones(m,1) x];

%Intialize theta to be a n+1 dimension vector with all elements equal to zero
theta = zeros(size(X,2),1);

costFunction(X, y, theta);

%Setting variables for gradient descent
alpha = 0.00003;
num_iter = 200;


[theta, J_history]  = gradientDescent(X,y,alpha,num_iter,theta);
hold on;
%Checking the fit of model against the scatter plot of the data
plot(X(:,2),X*theta,'-');
legend('Training Data', 'Linear Regression Model');
hold off
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
