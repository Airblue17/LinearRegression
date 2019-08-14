function [theta, J_history] = gradientDescent(X,y,alpha,num_iter,theta)

m = length(y);
J_history = zeros(num_iter,1);

for iter = 1:num_iter

delta = zeros(size(theta));
for i = 1:m,
delta = delta + ( theta' * X(i,:)' - y(i) ) * X(i,:)'; 
end;

theta = theta - alpha * (1/m) * delta;


J_history(iter) = costFunction(X, y, theta);

end

end
