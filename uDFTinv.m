function x=uDFTinv(X)
% inverse unitary DFT

N=length(X);
x=sqrt(N)*ifft(X);
end