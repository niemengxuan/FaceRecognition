%测试阶段

%直接以最邻近的类作为输出
accuracy = nearest(pcaTrainFace, trainLabels1, pcaTestFace, testLabels1);
fprintf("nearest accuracy : %%%.2f\n", accuracy*100);

%简单BP神经网络
num_iter = 10000;
alph = 0.01;
hlayer = 60;
par1 = [num_iter, alph];
par2 = [num_iter, alph, hlayer];
%无隐藏层
accuracy = nn(pcaTrainFace', trainLabels2', pcaTestFace', testLabels2', par1);
fprintf("nn accuracy : %%%.2f\n", accuracy*100);
%加一层隐藏层
accuracy = nn2(pcaTrainFace', trainLabels2', pcaTestFace', testLabels2', par2);
fprintf("nn2 accuracy : %%%.2f\n", accuracy*100);

function accuracy = nearest(pcaTrainFace, trainLabels, pcaTestFace, testLabels)
m = size(pcaTestFace, 1);    %取距离最近的训练数据比较，即k=1时的knn
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
