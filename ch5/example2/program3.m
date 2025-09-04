% p3.m - band-limited interpolation  
h = 1; xmax = 10; clf
x = -xmax:h:xmax; % computational grid 
xx = -xmax-h/20:h/10:xmax+h/20; % plotting grid 

figure('Position',[100 100 800 500]); % 调大图像窗口

for plt = 1:3
    subplot(3,1,plt)
    switch plt
        case 1, v = (x==0); % delta function
        case 2, v = (abs(x)<=3); % square wave
        case 3, v = max(0,1-abs(x)/3); % hat function
    end
    plot(x,v,'.','markersize',14), grid on
    p = zeros(size(xx));
    for i = 1:length(x)
        p = p + v(i)*sin(pi*(xx-x(i))/h)./(pi*(xx-x(i))/h);
    end
    line(xx,p,'linewidth',.7), axis([-xmax xmax -.5 1.5])
    set(gca,'xtick',[],'ytick',[0 1])
    set(gca,'FontName','Times New Roman','FontSize',20)
end

% 设置整体标题
sgtitle('Band-limited Interpolation','FontName','Times New Roman','FontSize',20)
saveas(gcf,'p3.png')