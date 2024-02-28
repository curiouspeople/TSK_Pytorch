function [RMSEtrain,RMSEtest,C,Sigma,W,yPredTest,C0,Sigma0,W0]=...
    MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,rr,P,nRules,nIt,Nbs,C0,Sigma0,W0)

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
Nbs=min(N,Nbs);
if nargin<11
    W0=zeros(nRules,M+1); % Rule consequents
    % FCM initialization
    [C0,U] = fcm(XTrain,nRules,[2 100 0.001 0]);
    Sigma0=C0;
    for r=1:nRules
        Sigma0(r,:)=std(XTrain,U(r,:));
        W0(r,1)=U(r,:)*yTrain/sum(U(r,:));
    end
    Sigma0(Sigma0==0)=mean(Sigma0(:));
end
C=C0; Sigma=Sigma0; W=W0;
minSigma=.1*min(Sigma0(:));

%% Iterative update
RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaC=zeros(nRules,M); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=ones(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    
    for n=1:Nbs
        idsKeep=rand(1,nRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:nRules
            if idsKeep(r)
                f(n,r)=prod(exp(-(XTrain(idsTrain(n),:)-C(r,:)).^2./(2*Sigma(r,:).^2)));
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=1;
            for r=1:nRules
                if idsKeep(r)
                    f(n,r)=prod(exp(-(XTrain(idsTrain(n),:)-C(r,:)).^2./(2*Sigma(r,:).^2)));
                end
            end
            idsKeep=true(1,nRules);
        end
        fBar=f(n,:)/sum(f(n,:));
        yR=[1 XTrain(idsTrain(n),:)]*W';
        yPred(n)=fBar*yR'; % prediction
        if isnan(yPred(n))
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        
        % Compute delta
        for r=1:nRules
            if idsKeep(r)
                temp=(yPred(n)-yTrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    % delta of c, sigma, and b
                    for m=1:M
                        deltaC(r,m)=deltaC(r,m)+temp*(XTrain(idsTrain(n),m)-C(r,m))/Sigma(r,m)^2;
                        deltaSigma(r,m)=deltaSigma(r,m)+temp*(XTrain(idsTrain(n),m)-C(r,m))^2/Sigma(r,m)^3;
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
    
    % Training RMSE on the minibatch
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    f=ones(NTest,nRules); % firing level of rules
    for n=1:NTest
        for r=1:nRules
            f(n,r)= prod(exp(-(XTest(n,:)-C(r,:)).^2./(2*Sigma(r,:).^2)));
        end
    end
    yR=[ones(NTest,1) XTest]*W';
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
    RMSEtest(it)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
    if isnan(RMSEtest(it)) && it>1
        RMSEtest(it)=RMSEtest(it-1);
    end
end
