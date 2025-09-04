clear all;close all;
L=20;N=128;
x=L/N*[-N/2:N/2-1];
k=2*pi/L*[-N/2:N/2-1];
u=sinc(x).^2;
ut=fft(u);
subplot(2,2,1)
plot(x,u,'k','LineWidth',1.5),xlabel x, ylabel u=sinc^2(x)
subplot(2,2,2)
plot(abs(ut),'k','LineWidth',1.5)
axis([1 N 0 7]), set(gca,'xtick',[]),ylabel abs(fft(u))
subplot(2,2,3)
plot(k,abs(fftshift(ut)),'k','LineWidth',1.5)
axis([-20 20 0 7]), xlabel k, ylabel abs(fftshift(fft(u)))
subplot(2,2,4)
plot(x,ifft(ifftshift(fftshift(ut))),'k','LineWidth',1.5)
xlabel x, ifft(ifftshift(fftshift(ut)))