function [pcaFace, V] = fastPCA(A, k)
mA = mean(A);
Z = A - mA;  %中心化
convMat = Z * Z'; %m×m
[V, ~] = eigs(convMat, k);  %m×k
V = Z' * V;   %n×k

for i = 1:k   %化为单位特征向量
    V(:,i)=V(:,i)/norm(V(:,i));
end
pcaFace = Z*V;  %m×k
end
