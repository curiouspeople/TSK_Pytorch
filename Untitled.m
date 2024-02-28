%% 文件名读取
datafileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\Raw_Data');
dirOutput=dir(fullfile(datafileFolder,'*.mat'));
dataFileNames={dirOutput.name};
LabelfileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\perclos_labels');
LabelOutput=dir(fullfile(LabelfileFolder,'*.mat'));
LabelFileNames={LabelOutput.name};

%% 获取数据
data=zeros(885,27200);
for i=1:23
    new_temp=load(strcat('D:\PyProject\深度学习\SEED-VIG\Raw_Data\',dataFileNames{i})).EEG.data;
    new_temp=reshape(new_temp,885,27200);
    data=vertcat(new_temp,data);
end
data=data(1:20355,:);
X=data;

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
[N0,~]=size(X);
N=round(N0*.7);%0.7作为训练集
idsTrain=datasample(1:N0,N,'replace',false);%随机采样下标，数目为1-350，且不重复
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];%构造数据集
label=zeros(885,1);
for i=1:23
    temp=load(strcat('D:\PyProject\深度学习\SEED-VIG\perclos_labels\', LabelFileNames{i})).perclos;
    label=vertcat(temp,label);
end
label=label(1:20355,:);
for k=1:20355
    if label(k)>0.35
       label(k)=2;
    else
       label(k)=1;
    end
end
yTrain=y(idsTrain);%训练集采样
yTest=y; yTest(idsTrain)=[];%测试集采样
[EntropyTrain,AccTest,BestPred,f,B,beta,gamma,fBar]=MBGD_CBS(XTrain,yTrain,XTest,yTest,alpha,P,eta,lambda,nRules,nIt,Nbs);
%% Plot results
figure('Position', get(0, 'Screensize'));
subplot(121);
plot(EntropyTrain,'linewidth',1);
xlabel('Iteration'); ylabel('Training cross-entropy');
subplot(122);
plot(AccTest,'linewidth',2);
set(gca,'yscale','log');
xlabel('Iteration'); ylabel('Test accuracy');



