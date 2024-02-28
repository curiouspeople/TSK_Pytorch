%% Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.
%% By Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all; %rng(0);

nMFs=5; % number of MFs in each input domain, used in MBGD_RDA and MBGD_RDA_T
alpha=.01; % initial learning rate
lambda=0.05; % L2 regularization coefficient
P=0.5; % DropRule rate
nIt=500; % number of iterations
Nbs=64; % batch size

temp=load('D:\PyProject\深度学习\SEED-VIG\Raw_Data\1_20151124_noon_2.mat');
data=temp.EEG.data; 
X=reshape(data,885,27200);

[coeff,score,latent,tsquared,explained,mu] = pca(X);

%1设定方差贡献度为99%
for i=1:length(explained)
	if sum(explained(1:i))>99
		an=i;
		break;
	end
end
X=score(:,1:an);
X = zscore(X);
%% X=fcm(X);
[C0,U,Obj]=fcm(X,an);
X=(C0*U)';

y=load('D:\PyProject\深度学习\SEED-VIG\perclos_labels\1_20151124_noon_2.mat');
y=y.perclos;
[N0,M]=size(X);%X归一化，
N=round(N0*.7);%N=350
idsTrain=datasample(1:N0,N,'replace',false);%随机采样下标，数目为1-350，且不重复
XTrain=X(idsTrain,:); yTrain=y(idsTrain);%训练集采样
XTest=X; XTest(idsTrain,:)=[];
yTest=y; yTest(idsTrain)=[];%测试集采样

%% 可以直接替换为python的train_test_split函数使用
RMSEtrain=zeros(6,nIt); RMSEtest=RMSEtrain;%均方根误差数组，初始化为0 
% Specify the total number of rules; use the original features without dimensionality reduction

[RMSEtrain(1,:),RMSEtest(1,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Gaussian MFs
[RMSEtrain(2,:),RMSEtest(2,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Trapezoidal MFs

% Dimensionality reduction, as in the following paper:
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
% 
% maxFeatures=5; % maximum number of features to use
% if M>maxFeatures
%     [~,XPCA,latent]=pca(X);%降维操作，latent是主成分得分
%     realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');%查找与非零元素对应的前 1 个索引。
%     usedDim=min(maxFeatures,realDim98);%返回最小维度
%     X=XPCA(:,1:usedDim); [N0,M]=size(X);%降维
% end
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];%构造数据集

% Specify the number of MFs in each input domain
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
[RMSEtrain(3,:),RMSEtest(3,:)]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Gaussian MFs
[RMSEtrain(4,:),RMSEtest(4,:),A,B,C,D]=MBGD_RDA_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Trapezoidal MFs

% Specify the total number of rules; each rule uses different membership functions
nRules=nMFs^M; % number of rules
[RMSEtrain(5,:),RMSEtest(5,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
[RMSEtrain(6,:),RMSEtest(6,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

%% Plot results
figure('Position', get(0, 'Screensize')); hold on;
linestyles={'k--','k-','g--','g-','b--','b-','r--','r-','m--','m-','c--','c-'};
for i=1:size(RMSEtrain,1)
    plot(RMSEtrain(i,:),linestyles{2*i-1},'linewidth',1);
    plot(RMSEtest(i,:),linestyles{2*i},'linewidth',2);
end
legend('Training RMSE1, RDA2 Gaussian','Test RMSE1, RDA2 Gaussian',...
    'Training RMSE2, RDA2 Trapezoidal','Test RMSE2, RDA2 Trapezoidal',...
    'Training RMSE3, RDA Gaussian','Test RMSE3, RDA Gaussian',...
    'Training RMSE4, RDA Trapezoidal','Test RMSE4, RDA Trapezoidal',...
    'Training RMSE5, RDA2 Gaussian','Test RMSE5, RDA2 Gaussian',...
    'Training RMSE6, RDA2 Trapezoidal','Test RMSE6, RDA2 Trapezoidal',...
    'location','eastoutside');
xlabel('Iteration'); ylabel('RMSE'); axis tight;

%% Plot the trapezoidal MFs obtained from MBGD_RDA_T
figure('Position', get(0, 'Screensize'));
for m=1:M 
    subplot(M,1,m); hold on;
    for n=1:nMFs
        plot([A(m,n) B(m,n) C(m,n) D(m,n)],[0 1 1 0],'linewidth',2);
    end
end
