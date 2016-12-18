clc;
clear;
close all;

%��������
fid = fopen('../iris.data');
trainingData = textscan(fid, '%f%f%f%f%s', 'delimiter', ',');
N = size(trainingData{1,1}, 1);
x = [trainingData{1,1}, trainingData{1,2}]';
y = [zeros(N/2,1); ones(N/2,1)];

%����չʾ
figure;
p1 = plot(x(1,1:N/2),x(2,1:N/2),'rx');
hold on;
p2 = plot(x(1,N/2+1:N),x(2,N/2+1:N),'g+');

%ѵ������
[w,b,cost] = Train(x,y,1e-6,2000);
fprintf('training finished');

%����ֱ��
hold on;
% l1 = ezplot(@(x,y) (1/(1+exp(w(1:2)'*[x;y]+b))-0.5),[4,7,2,4.5]);
l1 = ezplot(@(x,y) (w(1:2)'*[x;y]+b),[4,7,2,4.5]);

title('classiication result');
%xlabel('X');
%ylabel('Y');
legend([p1,p2,l1], 'iris-setosa','iris-versicolor','classificationLine');