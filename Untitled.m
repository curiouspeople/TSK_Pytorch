%% �ļ�����ȡ
datafileFolder=fullfile('D:\PyProject\���ѧϰ\SEED-VIG\Raw_Data');
dirOutput=dir(fullfile(datafileFolder,'*.mat'));
dataFileNames={dirOutput.name};
LabelfileFolder=fullfile('D:\PyProject\���ѧϰ\SEED-VIG\perclos_labels');
LabelOutput=dir(fullfile(LabelfileFolder,'*.mat'));
LabelFileNames={LabelOutput.name};

%% ��ȡ����
data=zeros(885,27200);
for i=1:23
    new_temp=load(strcat('D:\PyProject\���ѧϰ\SEED-VIG\Raw_Data\',dataFileNames{i})).EEG.data;
    new_temp=reshape(new_temp,885,27200);
    data=vertcat(new_temp,data);
end
data=data(1:20355,:);
X=data;

[coeff,score,latent,tsquared,explained,mu] = pca(X);

%1�趨����׶�Ϊ99%
for i=1:length(explained)
	if sum(explained(1:i))>99
		an=i;
		break;
	end
end
X=score(:,1:an);

X = zscore(X);
[N0,~]=size(X);
N=round(N0*.7);%0.7��Ϊѵ����
idsTrain=datasample(1:N0,N,'replace',false);%��������±꣬��ĿΪ1-350���Ҳ��ظ�
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];%�������ݼ�
label=zeros(885,1);
for i=1:23
    temp=load(strcat('D:\PyProject\���ѧϰ\SEED-VIG\perclos_labels\', LabelFileNames{i})).perclos;
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
yTrain=y(idsTrain);%ѵ��������
yTest=y; yTest(idsTrain)=[];%���Լ�����
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



