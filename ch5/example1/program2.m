% p2.m - convergence of periodic spectral method (compare p1.m)  
e = exp(1);

% 创建大尺寸 figure
figure('Color','w','Position',[100 100 700 700])
clf

% 坐标轴占满整个图像
figure('Color','w','Position',[100 100 700 700])
ax = axes(); % 左下角x,y,width,height

for N = 2:2:100
    h = 2*pi/N; 
    x = -pi + (1:N)'*h; 
    u = exp(sin(x)); 
    uprime = cos(x).*u;  

    % Construct spectral differentiation matrix
    column = [0 .5*(-1).^(1:N-1).*cot((1:N-1)*h/2)]; 
    D = toeplitz(column,column([1 N:-1:2]));  

    % Compute error
    error = norm(D*u-uprime,inf); 
    loglog(N,error,'.','markersize',15), hold on
end

grid on
xlabel('N','FontSize',20,'FontName','Times New Roman')
ylabel('error','FontSize',20,'FontName','Times New Roman')
title('Convergence of spectral differentiation','FontSize',20,'FontName','Times New Roman')

set(ax,'FontSize',20,'FontName','Times New Roman')
axis square

% 保存高清 PNG，占满画布内容
exportgraphics(gcf,'p2.png','Resolution',600)