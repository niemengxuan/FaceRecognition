function accuracy = nn2(trainSet, trainLabels, testSet, testLabels, par)
num_iter = par(1);
alph = par(2);
hlayer = par(3);
[W1, b1, W2, b2] = training(trainSet, trainLabels, num_iter, alph, hlayer);
accuracy = test(W1, b1, W2, b2, testSet, testLabels);
end

%训练
function [W1, b1, W2, b2] = training(X, Y, num_iter, alph, hlayer)
[n0, m] = size(X);
n1 = size(Y, 1);
W1 = rand(hlayer, n0);
b1 = rand(hlayer, 1);
W2 = rand(n1, hlayer);
b2 = rand(n1, 1);

batch_size = 50;        
for i = 1:num_iter       %梯度下降
    L = randperm(m);
    X_ = X(:, L(1:batch_size));
    Y_ = Y(:, L(1:batch_size));
    Z1 = W1 * X_ + b1;
    A1 = relu(Z1);
    A2 = sigmoid(W2 * A1 + b2);    
    [dW1, db1, dW2, db2] = grad(X_, A1, A2, Y_, W2, Z1);
    W2 = W2 - alph * dW2;
    b2 = b2 - alph * db2;
    W1 = W1 - alph * dW1;
    b1 = b1 - alph * db1;
end
end

function [dW1, db1, dW2, db2] = grad(X, A1, A2, Y, W2, Z1)
%各矩阵的维度
%X[nx, m]  A1[n1, m]  A2,Y[n2, m]  W1[n1, nx] W2[n2, n1] 
% b1[n1, 1]  b2[n2, 1]
m = size(X, 2);   
dZ2 = A2 - Y;  %[n2, m]
dW2 = 1 / m * dZ2 * A1'; %[n2, n1]
db2 = 1 / m * sum(dZ2, 2); %[n2, 1]
dZ1 = W2' * dZ2 .* (Z1>0); %[n1, m]
dW1 = 1 / m * dZ1 * X';  %[n1, nx]
db1 = 1 / m * sum(dZ1, 2);  %[n1, 1]
end


function s = relu(z)
s = max(0, z);
end
function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function accuracy = test(W1, b1, W2, b2, X, Y) %计算测试集正确率
m = size(Y, 2);
A = relu(W1 * X + b1);
Y1 = sigmoid(W2 * A + b2);
Y1(Y1 >= 0.5) = 1;
Y1(Y1 < 0.5) = 0;
D = (Y1 == Y);
d = sum(~D);
accuracy = sum(d == 0) / m;
end