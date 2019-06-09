function [trainFace, trainLabels, testFace, testLabels] = loadFace(l)
if nargin < 1
    l = 0.6;  %Ĭ��ȡѵ����������64��
end
l = round(10*l);
tr_num = l*40;   %ѵ����������
te_num = (10-l)*40;  %������������
[m, n] = size(imread('ORL/s1/1.pgm'));
trainFace = zeros(tr_num, m*n);    %Ԥ��������洢���ݣ����д洢��������
trainLabels = zeros(tr_num, 1);
testFace = zeros(te_num, m*n);
testLabels = zeros(te_num, 1);

for i = 1:40               %��ȡORL�е���������
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
% index = randperm(tr_num);    %����ѵ����,batch����Բ���Ҫ�˲���
% trainFace(:, :) = trainFace(index, :);
% trainLabels(:, :) = trainLabels(index, :);
end