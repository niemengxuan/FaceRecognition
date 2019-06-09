function [pcaFace, V] = fastPCA(A, k)
mA = mean(A);
Z = A - mA;  %���Ļ�
convMat = Z * Z'; %m��m
[V, ~] = eigs(convMat, k);  %m��k
V = Z' * V;   %n��k

for i = 1:k   %��Ϊ��λ��������
    V(:,i)=V(:,i)/norm(V(:,i));
end
pcaFace = Z*V;  %m��k
end
