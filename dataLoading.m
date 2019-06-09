%׼������

clear; clc;
[trainFace, trainLabels1, testFace, testLabels1] = loadFace(); %��ȡ����
[trainLabels2, testLabels2] = reform(trainLabels1, testLabels1); %ת�������ʽ
picShow(trainFace(1:20, :)')  %��ʾ����ѵ��������
[pcaFace, V] = fastPCA(trainFace, 30);   %��ȡ������pca��ά
picShow(V)   %��ʾ������
pcaTrainFace = scaling(pcaFace);  %��һ��ѵ����
pcaTestFace = testFace * V;      %��V����Բ��Լ����н�ά����
pcaTestFace = scaling(pcaTestFace);  %��һ�����Լ�

function picShow(V)  %��ʾ����
figure
img = zeros(112, 92);
for i = 1:20
    subplot(4, 5, i)
    img(:) = V(:, i);
    imshow(img, [])
end
end

function scaledFace = scaling(inputFace)   %��һ�����������ŵ���0��1��������
range = minmax(inputFace');
range = range';
minVec = range(1, :);
maxVec = range(2, :);
scaledFace = (inputFace - minVec(1,:))./(maxVec(1,:) - minVec(1,:));
end

%ת����ǩ�ĸ�ʽ�����ָ�ʽӦ��������ʱ������
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