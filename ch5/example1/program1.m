% p1.m - convergence of fourth-order finite differences
e = exp(1);
Nvec = 2.^(3:12);

% 创建大尺寸 figure (单位：像素)
figure('Color','w','Position',[100 100 700 700])
ax = axes(); % 左下角x,y,width,height

for N = Nvec
    h = 2*pi/N; 
    x = -pi + (1:N)'*h; 
    u = exp(sin(x)); 
    uprime = cos(x).*u;  

    % Construct sparse 4th-order differentiation matrix
    e1 = ones(N,1); 
    D = sparse(1:N,[2:N 1],2*e1/3,N,N) - sparse(1:N,[3:N 1 2],e1/12,N,N); 
    D = (D-D')/h;  

    % Compute error
    error = norm(D*u-uprime,inf); 
    loglog(N,error,'.','markersize',15), hold on 
end

grid on
xlabel('N','FontSize',20,'FontName','Times New Roman')
ylabel('error','FontSize',20,'FontName','Times New Roman')
title('Convergence of 4th-order finite differences','FontSize',20,'FontName','Times New Roman')
semilogy(Nvec,Nvec.^(-4),'--')
text(105,5e-8,'N^{-4}','FontSize',20,'FontName','Times New Roman')

set(ax,'FontSize',20,'FontName','Times New Roman') 
axis square % 保持坐标轴为正方形比例

% 保存高清图片，占满画布
exportgraphics(gcf,'p1.png','Resolution',300)
