function [trainFace, trainLabels, testFace, testLabels] = loadFace(l)
if nargin < 1
    l = 0.6;  %默认取训练测试数据64分
end
l = round(10*l);
tr_num = l*40;   %训练样本数量
te_num = (10-l)*40;  %测试样本数量
[m, n] = size(imread('ORL/s1/1.pgm'));
trainFace = zeros(tr_num, m*n);    %预定义变量存储数据，按行存储人脸数据
trainLabels = zeros(tr_num, 1);
testFace = zeros(te_num, m*n);
testLabels = zeros(te_num, 1);

for i = 1:40               %读取ORL中的人脸数据
    ipath = strcat('ORL/s', num2str(i), '/');
    for j = 1:10
        path = strcat(ipath, num2str(j), '.pgm');
        if j <= l            
            img = imread(path);
            trainFace((i-1)*l+j, :) = img(:);
            trainLabels((i-1)*l+j) = i;
            
        else
            img = imread(path);
            testFace((i-1)*(10-l)+j-l, :) = img(:);
            testLabels((i-1)*(10-l)+j-l) = i;
        end
    end
end
% index = randperm(tr_num);    %打乱训练集,batch后可以不需要此步骤
% trainFace(:, :) = trainFace(index, :);
% trainLabels(:, :) = trainLabels(index, :);
end