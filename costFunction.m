function J = costFunction(X, y, theta)

%no of training examples
m = length(y);

J = 0;

for i = 1:m,

J = J + ( (theta' * X(i,:)') - y(i) )^2;

end


J = (1/(2*m))*J;

end