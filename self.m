clear;
clc; clearvars; close all; %rng(0);
t1=clock;
  %代码段


alpha=.08; % initial learning rate

eta=0.3; % L2 regularization coefficient 0.5
P=0.3; % DropRule rate
nIt=500; % number of iterations
Nbs=128; % batch size
nRules=5;%fuzy rules number

split=0.7;%训练集与测试集划分


temp=load('D:\PyProject\深度学习\SEED-VIG\Raw_Data\1_20151124_noon_2.mat');
data=temp.EEG.data;
% data=data(:,1:38766);

X=reshape(data,885,27200);
% X = zscore(X);
% X = gpuArray(X);
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
%% 
X = zscore(X);
[N0,~]=size(X);

N=round(N0*split);%0.7作为训练集
idsTrain=datasample(1:N0,N,'replace',false);%随机采样下标，数目为1-350，且不重复
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];%构造数据集

y=load('D:\PyProject\深度学习\SEED-VIG\perclos_labels\1_20151124_noon_2.mat');
%y=data(:,end); y=y-mean(y);%X 500*7，Y 500*1 
y=y.perclos;
%y=y-mean(y);
y=reshape(y,885,1);
% y = gpuArray(y);
% y=y(:,2);
% N=round(N0*.7);%N=350
idsTrain=datasample(1:N0,N,'replace',false);%随机采样下标，数目为1-350，且不重复
% for i=1:885
%     if y(i)>0.35
%         y(i)=2;
%     else
%         y(i)=1;
%     end
% end
yTrain=y(idsTrain);%训练集采样
yTest=y; yTest(idsTrain)=[];%测试集采样
RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;%均方根误差数组，初始化为0 
[RMSEtrain,RMSEtest,C,Sigma,W,yPredTest,C0,Sigma0,W0]=MBGD_BNDS(XTrain,yTrain,XTest,yTest,alpha,eta,P,nRules,nIt,Nbs);
t2=clock;
disp(etime(t2,t1));

% [EntropyTrain,AccTest,AccTrain,mStepSize,stdStepSize,BestPred,f,B,beta,gamma,fBar]=MBGD_CBS(XTrain,yTrain,XTest,yTest,alpha,P,eta,lambda,nRules,nIt,Nbs); % Gaussian MFs
% [EntropyTrain,AccTest]=MBGD_CBS(XTrain,yTrain,XTest,yTest,alpha,P,eta,lambda,nRules,nIt,Nbs); % Gaussian MFs
% [EntropyTrain,AccTest,BestPred,f,B,beta,gamma,fBar,M,Sigma]=MBGD_CBS(XTrain,yTrain,XTest,yTest,alpha,P,eta,lambda,nRules,nIt,Nbs);
% [EntropyTrain,AccTest,gamma,beta]=MBGD_UR_BN(XTrain,yTrain,XTest,yTest,alpha,eta,lambda,nRules,nIt,Nbs);
 %% 文件名读取
datafileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\Raw_Data');
dirOutput=dir(fullfile(datafileFolder,'*.mat'));
dataFileNames={dirOutput.name};
LabelfileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\perclos_labels');
LabelOutput=dir(fullfile(LabelfileFolder,'*.mat'));
LabelFileNames={LabelOutput.name};
% % %% 对每一个被试进行测试，保存准确率
RMSEcros=zeros(23,1);
for i=1:23
    disp('-------------------------------------------------------------')
    disp(i)
    disp('-------------------------------------------------------------')
    %对原始数据的处理
    cro_data=load(strcat('D:\PyProject\深度学习\SEED-VIG\Raw_Data\',dataFileNames{i})).EEG.data;
    temp=reshape(cro_data,885,27200);
    [coeff,score,latent,tsquared,explained,mu] = pca(temp);
    for j=1:length(explained)
        if sum(explained(1:j))>99
            an=j;
            break;
        end
    end
    cro_X=score(:,1:40);
    cro_X = zscore(cro_X);
    %% fcm
    [C0,U,Obj]=fcm(cro_X,40);
    cro_X=(C0*U)';
    cro_X = zscore(cro_X);
% 
    cro_X = zscore(cro_X); 
    %加载标签
    y_label=load(strcat('D:\PyProject\深度学习\SEED-VIG\perclos_labels\', LabelFileNames{i})).perclos;
    %开始计算精度
    cro_N=size(cro_X,1);

    f=ones(cro_N,nRules); % firing level of rules
    for n=1:cro_N
        for r=1:nRules
            x=(-(cro_X(n,:)-C(r,:)).^2./(2*Sigma(r,:).^2));
            z=x-max(x);
%             disp(size(z))
            x=softmax1(z,40);
            f(n,r)=prod(exp(1/(1+exp(-x'))));
%             f(n,r)= prod(exp(-(XTest(n,:)-C(r,:)).^2./(2*Sigma(r,:).^2)));
        end
    end
    yR=[ones(cro_N,1) cro_X]*W';
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
    RMSEcros(i)=sqrt((y_label-yPredTest)'*(y_label-yPredTest)/cro_N);
end

% %% Plot results
% figure('Position', get(0, 'Screensize'));
% subplot(121);
% plot(EntropyTrain,'linewidth',1);
% xlabel('Iteration'); ylabel('Training cross-entropy');
% subplot(122);
% plot(AccTest,'linewidth',2);
% set(gca,'yscale','log');
% xlabel('Iteration'); ylabel('Test accuracy');
%% GRAPH
% x=linspace(-0.3,0.3,100);
% y=zeros(100);
% y(45)=1;
% plot(x,y,'r')
% title('4st rule')
% ylabel('C5')
% x=ones(1,500);
% for i=1:500
%     x(i)=i;
% end
% 
% y1=RMSEcros;
% y2=RMSEtrain;
% plot(x,y1,'r')
% % hold on
% plot(x,y2,'b')
% ylabel('RMSE')
% xlabel('nIt')

% c5 4st y(71)=1;
% c5 3st y(40)=1;
% c5 2st y(22)=1;
% c5 1st y(54)=1;
% c4 5st y(37)=1;
% c4 4st y(62)=1;
% c4 3st y(54)=1;
% c4 2st y(24)=1;
% c4 1st y(47)=1;
% c3 5st y(61)=1;
% c3 4st y(53)=1;
% c3 3st y(61)=1;
% c3 2st y(35)=1;
% c3 1st y(53)=1;
% c2 1st y(50)=1;
% c2 2st y(35)=1;
% c2 3st y(44)=1;
% c2 4st y(63)=1;
% c2 5st y(44)=1;