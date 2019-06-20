%准备数据

clear; clc;
[trainFace, trainLabels1, testFace, testLabels1] = loadFace(0.7); %读取数据
[trainLabels2, testLabels2] = reform(trainLabels1, testLabels1); %转换输出格式
picShow(trainFace(1:20, :)')  %显示部分训练集人脸
[pcaFace, V] = fastPCA(trainFace, 40);   %提取特征，pca降维
picShow(V)   %显示特征脸
pcaTrainFace = scaling(pcaFace);  %归一化训练集
pcaTestFace = testFace * V;      %由V矩阵对测试集进行降维操作
pcaTestFace = scaling(pcaTestFace);  %归一化测试集

function picShow(V)  %显示函数
figure
img = zeros(112, 92);
for i = 1:20
    subplot(4, 5, i)
    img(:) = V(:, i);
    imshow(img, [])
end
end

function scaledFace = scaling(inputFace)   %归一化函数，缩放到【0，1】区间内
range = minmax(inputFace');
range = range';
minVec = range(1, :);
maxVec = range(2, :);
scaledFace = (inputFace - minVec(1,:))./(maxVec(1,:) - minVec(1,:));
end

%转换标签的格式，这种格式应用神经网络时更方便
function [trainLabels2, testLabels2] = reform(trainLabels1, testLabels1)
n1 = size(trainLabels1, 1);
n2 = size(testLabels1, 1);
trainLabels2 = zeros(n1, 40);
testLabels2 = zeros(n2, 40);
for i = 1:n1
    trainLabels2(i, trainLabels1(i)) = 1;
end
for i = 1:n2
    testLabels2(i, testLabels1(i)) = 1;
end
end
