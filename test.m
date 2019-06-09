%���Խ׶�

%ֱ�������ڽ�������Ϊ���
accuracy = nearest(pcaTrainFace, trainLabels1, pcaTestFace, testLabels1);
fprintf("nearest accuracy : %%%.2f\n", accuracy*100);

%��BP������
num_iter = 100000;
alph = 0.005;
hlayer = 40;
par1 = [num_iter, alph];
par2 = [num_iter, alph, hlayer];
%�����ز�
accuracy = nn(pcaTrainFace', trainLabels2', pcaTestFace', testLabels2', par1);
fprintf("nn accuracy : %%%.2f\n", accuracy*100);
%��һ�����ز�
accuracy = nn2(pcaTrainFace', trainLabels2', pcaTestFace', testLabels2', par2);
fprintf("nn2 accuracy : %%%.2f\n", accuracy*100);

function accuracy = nearest(pcaTrainFace, trainLabels, pcaTestFace, testLabels)
m = size(pcaTestFace, 1);    %ȡ���������ѵ�����ݱȽϣ���k=1ʱ��knn
n = size(pcaTrainFace, 1);
count = 0;
for i = 1:m
    z = pcaTrainFace - repmat(pcaTestFace(i,:),n,1);
    dist = sum(z.^2, 2);    
    [~, index] = sort(dist);  
    if trainLabels(index(1)) == testLabels(i)
        count = count + 1;
    end
end
accuracy = count/m;
end
