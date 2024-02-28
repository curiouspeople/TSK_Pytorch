%% Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.
%% By Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all; %rng(0);

nMFs=2; % number of MFs in each input domain, used in MBGD_RDA and MBGD_RDA_T
alpha=.01; % initial learning rate
lambda=0.05; % L2 regularization coefficient
P=0.5; % DropRule rate
nIt=500; % number of iterations
Nbs=64; % batch size
temp=load('D:\PyProject\深度学习\SEED-VIG\Raw_Data\1_20151124_noon_2.mat');
%temp=load('NO2.mat');

%data=temp.data;
data=temp.EEG.data;
%X=data(:,1:end-1); 
X=reshape(data,885,27200);
X = zscore(X);
% X=normalize(X);

y=load('D:\PyProject\深度学习\SEED-VIG\perclos_labels\1_20151124_noon_2.mat');
%y=data(:,end); y=y-mean(y);%X 500*7，Y 500*1 
y=y.perclos;
%y=y-mean(y);
[center,U,obj_fun] = fcm(X,8);
X=U'*center;
y=reshape(y,885,1);
for i=1:885
    if y(i)>0.35
        y(i)=1;
    else
        y(i)=0;
    end
end
[N0,M]=size(X);%X归一化，
N=round(N0*.7);%N=350
idsTrain=datasample(1:N0,N,'replace',false);%随机采样下标，数目为1-350，且不重复
XTrain=X(idsTrain,:); yTrain=y(idsTrain);%训练集采样
XTest=X; XTest(idsTrain,:)=[];
yTest=y; yTest(idsTrain)=[];%测试集采样

%% 可以直接替换为python的train_test_split函数使用
RMSEtrain=zeros(6,nIt); RMSEtest=RMSEtrain;%均方根误差数组，初始化为0 

% Specify the total number of rules; use the original features without dimensionality reduction
nRules=30; % number of rules, used in MBGD_RDA2 and MBGD_RDA2_T
% [RMSEtrain(1,:),RMSEtest(1,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(2,:),RMSEtest(2,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

% Dimensionality reduction, as in the following paper:
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.

maxFeatures=5; % maximum number of features to use
if M>maxFeatures
    [~,XPCA,latent]=pca(X);%降维操作，latent是主成分得分
    realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');%查找与非零元素对应的前 1 个索引。
    usedDim=min(maxFeatures,realDim98);%返回最小维度
    X=XPCA(:,1:usedDim); [N0,M]=size(X);%降维
end
XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];%构造数据集

% Specify the number of MFs in each input domain
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
[RMSEtrain(3,:),RMSEtest(3,:)]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(4,:),RMSEtest(4,:),A,B,C,D]=MBGD_RDA_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Trapezoidal MFs

% Specify the total number of rules; each rule uses different membership functions
nRules=nMFs^M; % number of rules
% [RMSEtrain(5,:),RMSEtest(5,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(6,:),RMSEtest(6,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs
RMSEtest(isnan(RMSEtest)) = 0;
RMSEtest(isinf(RMSEtest)) = 0;
RMSEtrain(isnan(RMSEtrain)) = 0;
RMSEtrain(isinf(RMSEtrain)) = 0;

%% Plot the trapezoidal MFs obtained from MBGD_RDA_T
% figure('Position', get(0, 'Screensize'));
% for m=1:M
%     subplot(M,1,m); hold on;
%     for n=1:nMFs
%         plot([A(m,n) B(m,n) C(m,n) D(m,n)],[0 1 1 0],'linewidth',2);
%     end
% end


% This function implements the MBGD-RDA algorithm in the following paper:
%
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
%
% It specifies the number of Gaussian MFs in each input domain by nMFs.
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
%
% By Dongrui Wu, drwu@hust.edu.cn
%
% %% Inputs:
% XTrain: N*M matrix of the training inputs. N is the number of samples, and M the feature dimensionality.
% yTrain: N*1 vector of the labels for XTrain
% XTest: NTest*M matrix of the test inputs
% yTest: NTest*1 vector of the labels for XTest
% alpha: scalar, learning rate
% rr: scalar, L2 regularization coefficient 
% P: scalar in [0.5, 1), dropRule rate
% nMFs: scalar in [2, 5], number of MFs in each input domain
% nIt: scalar, maximum number of iterations
% Nbs: batch size. typically 32 or 64
%
% %% Outputs:
% RMSEtrain: 1*nIt vector of the training RMSE at different iterations
% RMSEtest: 1*nIt vector of the test RMSE at different iterations
% C: M*nMFs matrix of the centers of the Gaussian MFs
% Sigma: M*nMFs matrix of the standard deviations of the Gaussian MFs
% W: nRules*(M+1) matrix of the consequent parameters for the rules. nRules=nMFs^M.

beta1=0.9; beta2=0.999;
rr=lambda
[N,M]=size(XTrain); NTest=size(XTest,1);%
if Nbs>N; Nbs=N; end
nMFsVec=nMFs*ones(M,1);% [2,2,2,2,2]
nRules=nMFs^M; % number of rules
C=zeros(M,nMFs); Sigma=C; %C 5*2
W=zeros(nRules,M+1);%c 2*5 序列参数,br
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),nMFs);%生成 n 个点。这些点的间距为 (max-min)/(n-1)。，聚类中心
    Sigma(m,:)=std(XTrain(:,m));%标准差
end
minSigma=min(Sigma(:));
%% Iterative update
mu=zeros(M,nMFs);  RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
%初始化
for it=1:nIt%k=1->K
    deltaC=zeros(M,nMFs); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=ones(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);%随机选择一个Nbs大小的数据训练，
    idsGoodTrain=true(Nbs,1);
    
    for n=1:Nbs
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        %数据模糊化
        idsKeep=rand(1,nRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:nRules
            if idsKeep(r)
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        %模糊规则剪切
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=true(1,nRules);
            f(n,:)=1;
            for r=1:nRules
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        
        fBar=f(n,:)/sum(f(n,:));% normalized firing levels 
        yR=[1 XTrain(idsTrain(n),:)]*W';
        yPred(n)=fBar*yR'; % prediction
        %TSK模糊系统到此则预测完毕，下面就是优化部分
        if isnan(yPred(n))%如果预测结果有问题，则跳过此次计算结果
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        
        % Compute delta
        for r=1:nRules
            if idsKeep(r)
                temp=(yPred(n)-yTrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    vec=idx2vec(r,nMFsVec);
                    % delta of c, sigma, and b
                    for m=1:M
                        deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                        deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                        deltaW(r,m+1)=deltaW(r,m+1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r)*XTrain(idsTrain(n),m);
                    end
                    % delta of b0
                    deltaW(r,1)=deltaW(r,1)+(yPred(n)-yTrain(idsTrain(n)))*fBar(r);
                end
            end
        end
    end
    
    % AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(minSigma,Sigma-lrSigma.*mSigmaHat);
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^it);
    vWHat=vW/(1-beta2^it);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    disp(max((yTrain)))
    disp(max(yPred))
    for i=1:size(yPred)
            if yPred(i)>0.35
                yPred(i)=1;
            else
                yPred(i)=0;
            end
     end 
    % Training RMSE
    RMSEtrain(it)=1-sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)))/64;
    %sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    f=ones(NTest,nRules); % firing level of rules
    for n=1:NTest
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        
        for r=1:nRules % firing levels of rules
            idsMFs=idx2vec(r,nMFsVec);
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));
            end
        end
    end
    yR=[ones(NTest,1) XTest]*W';%增加一列全部变为1 yr=1 x*w
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction f就是fr(x)
   

%     RMSEtest(it)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
     for i=1:size(yPredTest)
            if yPredTest(i)>0.35
                yPredTest(i)=1;
            else
                yPredTest(i)=0;
            end
     end
    RMSEtest(it)=1-sum((yTest-yPredTest))/64;
    if isnan(RMSEtest(it)) && it>1
        RMSEtest(it)=RMSEtest(it-1);
    end
end

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

function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector index of MFs
vec=zeros(1,length(nMFs));
prods=[1; cumprod(nMFs(end:-1:1))];
if idx>prods(end)
    error('Error: idx is larger than the number of rules.');
end
prev=0;
for i=1:length(nMFs)
    vec(i)=floor((idx-1-prev)/prods(end-i))+1;
    prev=prev+(vec(i)-1)*prods(end-i);
end
end

