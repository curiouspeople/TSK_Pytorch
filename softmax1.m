function softMax=softmax1(x,n)
    sum=0;
    softMax=nan(1,n);
    for k=1:n
      sum = sum+exp(x(k));
    end
    for m=1:n
        softMax(m)=exp(x(m))/sum;
    end
    
end