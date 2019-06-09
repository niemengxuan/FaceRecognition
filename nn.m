function accuracy = nn(trainSet, trainLabels, testSet, testLabels, par)
num_iter = par(1);
alph = par(2);
[W, b] = training(trainSet, trainLabels, num_iter, alph);
accuracy = test(W, b, testSet, testLabels);
end

function [W, b] = training(X, Y, num_iter, alph)
[n0, m] = size(X);
n1 = size(Y, 1);
W = rand(n1, n0);
b = rand(n1, 1);
batch_size = 100;

% X_ = X;
% Y_ = Y;
for i = 1:num_iter
    L = randperm(m);
    X_ = X(:, L(1:batch_size));
    Y_ = Y(:, L(1:batch_size));
    [dW, db] = propagate(W, b, X_, Y_);
    W = W - alph * dW;
    b = b - alph * db;
end

end

function [dW, db] = propagate(W, b, X, Y)
%X[n0, m]   Y[n1, m]  W[n1, n0]  b[n1, 1]
m = size(X, 2);   
A = sigmoid(W*X + b);  %n1¡Ám
dZ = A - Y;  %[n1, m]
dW = 1 / m * dZ * X'; %[n1, n0]
db = 1 / m * sum(dZ, 2); %[n1, 1]

end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function accuracy = test(W, b, X, Y)
m = size(Y, 2);
Y1 = sigmoid(W * X + b);
Y1(Y1>=0.5) = 1;
Y1(Y1 < 0.5) = 0;
D = (Y1 == Y);
d = sum(~D);
accuracy = sum(d == 0) / m;
end