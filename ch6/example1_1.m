clear all; close all;
L=1; h=0.05; x=0:h:L;
u_Euler=zeros(length(x),1); u_Euler(1)=3;
u_pc=u_Euler; u_ode45=u_Euler;

for n=1:length(x)-1
    % 欧拉法
    u_Euler(n+1)=u_Euler(n)+h*(-3*u_Euler(n)+6*x(n)+5);
    % 预测-校正法
    k1=h*(-3*u_pc(n)+6*x(n)+5);
    k2=h*(-3*(u_pc(n)+k1)+6*(x(n)+h)+5);
    u_pc(n+1)=u_pc(n)+(k1+k2)/2;
end

% ode45
[x,u_ode45]=ode45(@(x,u)[-3*u+6*x+5],x,u_ode45(1));

% 解析解
u_exact=2*exp(-3*x)+2*x+1;

% 解的比较图
figure('Position',[100 100 800 350])
plot(x,u_Euler,'xk',x,u_pc,'ok',x,u_ode45,'+k',x,u_exact,'k','MarkerSize',8,'LineWidth',1.5)
axis([0 1 2.3 3.15])
xlabel('x','FontName','Times New Roman','FontSize',12)
ylabel('u','FontName','Times New Roman','FontSize',12)
set(gca,'FontSize',12)
legend('欧拉法','预测-校正法','ode45','解析解','Location','North')
exportgraphics(gcf,'1.png','Resolution',300)   % 保存为1.png

% 三种算法的误差
Error(:,1)=abs(u_Euler-u_exact(:));
Error(:,2)=abs(u_pc-u_exact(:));
Error(:,3)=abs(u_ode45-u_exact(:));

% 误差图
figure('Position',[100 500 800 350])
plot(x,Error(:,1),'--ok','LineWidth',1.5); hold on
plot(x,Error(:,2),':^k','LineWidth',1.5);
plot(x,Error(:,3),'-sk','LineWidth',1.5);
xlabel('x','FontName','Times New Roman','FontSize',12)
ylabel('Abs error','FontName','Times New Roman','FontSize',12)
set(gca,'FontSize',12)
legend('欧拉法误差','预测-校正法误差','ode45误差','Location','NorthWest')
exportgraphics(gcf,'2.png','Resolution',300)   % 保存为2.png
