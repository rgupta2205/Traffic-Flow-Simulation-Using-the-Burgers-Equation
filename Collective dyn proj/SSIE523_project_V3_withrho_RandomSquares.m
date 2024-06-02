clc
close all
clear 
%%=========================================================================
% Geometry 
%%=========================================================================
Lx=2 %street x-direction 
Ly=6 %street y-direction
nx = 41; %no of steps in x-direction
ny = 41;
dx = Lx/(nx-1); %resolution in x-direction
dy = Ly/(ny-1);
x = linspace(0,Lx,nx);
y = linspace(0,Ly,ny);

% CLT condition (time and space resolution)
sigma = .01; %0.009
nu=1;
nt = 200%120; %no of steps in time
dt = sigma*dx*dy/nu; %time resolution 

%%=========================================================================
% Compute the velocity profile
%%=========================================================================
v_max=1;  % Maximum car speed (km/h)
rho_max=1; % Maximum car density (cars/km)
ratio=rho_max/v_max; %veocity-density relation    v = v_max * (1 - rho / rho_max);

rho0= 0.2 % set the background density 
u0 = v_max * (1 - rho0 / rho_max) %initial velocity related to the initial density 
u = u0*ones(ny,nx); %velocity profile in x-direction
v =0*ones(ny,nx)%velocity profile in y-direction (Assume zero to ignore calculations)

%%=========================================================================
%  Initial conditions (defining spots previous was the background IC)
%%=========================================================================
% Option 1 constant spot density
% rho= 0.7 % spots density
% u0 = v_max * (1 - rho / rho_max); %initial velocity related to the initial density 

%%% Select the density randomly
rho_vector = rho0:0.1:1;
u0_vector = v_max * (1 - rho_vector / rho_max); %initial velocity related to the initial density 


% Define the dimensions of the squares
square_size = 5;

% Define the number of squares to generate
num_squares = randi(30,1,1);

% Generate random locations for the squares
square_locations = randi([1, 40-square_size+1], num_squares, 2);

% Create the squares by setting the appropriate elements of the space to density value
for i = 1:num_squares
    random_index = randi(length(u0_vector));
    u(square_locations(i,1):(square_locations(i,1)+square_size-1), square_locations(i,2):(square_locations(i,2)+square_size-1)) =  u0_vector(random_index);
    u0_vector(random_index)
end

figure 
mesh(u); title ('velocity profile x-direction')
%%=========================================================================
%  IC Visualization
%%=========================================================================
% Plot the resulting squares
% imagesc(u);
% colormap(gray);

% figure
% subplot(121)
% rho= ones(size(u))-(ratio*u);%convert from velocity to density
% contourf(x,y,rho)
% xlabel('x-direction'); ylabel ('y-direction')
% title ('Density IC')
% % xline (x(Line),'--r','LineWidth',2)
% colorbar
% subplot(122)
% plot3(rho(:,2),rho(:,20),rho(:,40),'b')
% hold on
% plot3(rho(2,:),rho(20,:),rho(40,:),'r')
% xlabel('in'); ylabel('mid'); zlabel('out')
% legend ('x-direction ','y-direction')
% ylim([0,1]) 
rho= ones(size(u))-(ratio*u);%convert from velocity to density
figure 
subplot(131)
contourf(x,y,rho)
xlabel('x-direction'); ylabel ('y-direction')
title ('Density profile')
colorbar
c=['b','r','g','k'];
cnt=1;
for i=3:10:40
    subplot(132)
    hold on
    plot(x,rho(i,:),c(cnt))
    subplot(133)
    hold on
    plot(y,rho(:,i),c(cnt))
    subplot(131)
    xline(x(i),c(cnt)); yline(y(i),c(cnt))
    cnt=cnt+1;
end
subplot(132)
legend ('1st','2nd','3rd','4th')
title ('y-direction'); ylabel('density')
subplot(133)
legend ('1st','2nd','3rd','4th')
title ('x-direction'); ylabel('density')


%%=========================================================================
%  Run the Simulation for nt steps 
%%=========================================================================
[X, Y] = meshgrid(x,y);
% Burgers equation discretization in time and space using Finite difference
for n=1:nt+1
    for i=2:(ny-1)
        for j=2:(nx-1)
        u(i,j)=u(i,j)- (dt/dx) * u(i,j)*(u(i,j) -u(i-1,j)) - (dt/dy) * v(i,j)*(u(i,j)-u(i,j-1)) + (nu*dt/dx^2) *(u(i-1,j)-2*u(i,j)+u(i-1,j)) + (nu*dt/dy^2) * (u(i,j+1)-2*u(i,j)+u(i,j-1));
        v(i,j)=v(i,j)- (dt/dx) * u(i,j)*(v(i,j) -v(i-1,j)) - (dt/dy) * v(i,j)*(v(i,j)-v(i,j-1)) + (nu*dt/dx^2) *( v(i-1,j)-2*v(i,j)+v(i-1,j)) + (nu*dt/dy^2) * (v(i,j+1)-2*v(i,j)+v(i,j-1));
            u(1:ny,1)=1;
            u(1,1:nx)=1;
            u(ny,1:nx)=1;
            u(1:nx,ny)=1;
            v(1:ny,1)=1;
            v(1,1:nx)=1;
            v(ny,1:nx)=1;
            v(1:nx,ny)=1;
        end
    end
    
    rho= ones(size(u))-(ratio*u);
%     h1=subplot(121)
%     cla(h1);
%     rho= ones(size(u))-(ratio*u);%convert from velocity to density
%     contourf(x,y,rho)
%     xlabel('x-direction'); ylabel ('y-direction')
%     title ('Density profile')
%     % xline (x(Line),'--r','LineWidth',2)
%     colorbar
%     h2=subplot(122)
%     cla(h2);
%     plot3(rho(:,2),rho(:,20),rho(:,40),'b')
%     hold on
%     plot3(rho(2,:),rho(20,:),rho(40,:),'r')
%     xlabel('in'); ylabel('mid'); zlabel('out')
%     legend ('x-direction ','y-direction')
%     ylim([0,1]) 
%     drawnow
end

figure
subplot(131)
rho= ones(size(u))-(ratio*u);%convert from velocity to density
contourf(x,y,rho)
xlabel('x-direction'); ylabel ('y-direction')
title ('Density profile')
% xline (x(Line),'--r','LineWidth',2)
colorbar
c=['b','r','g','k'];
cnt=1;
for i=3:10:40
    subplot(132)
    hold on
    plot(x,rho(i,:),c(cnt))
    subplot(133)
    hold on
    plot(y,rho(:,i),c(cnt))
    subplot(131)
    xline(x(i),c(cnt)); yline(y(i),c(cnt))
    cnt=cnt+1;
end
subplot(132)
legend ('1st','2nd','3rd','4th')
title ('y-direction'); ylabel('density')
subplot(133)
legend ('1st','2nd','3rd','4th')
title ('x-direction'); ylabel('density')



% subplot(121)
% rho= ones(size(u))-(ratio*u);%convert from velocity to density
% contourf(x,y,rho)
% xlabel('x-direction'); ylabel ('y-direction')
% title ('Density profile')
% % xline (x(Line),'--r','LineWidth',2)
% colorbar
% subplot(122)
% plot3(rho(:,2),rho(:,20),rho(:,40),'b')
% hold on
% plot3(rho(2,:),rho(20,:),rho(40,:),'r')
% xlabel('in'); ylabel('mid'); zlabel('out')
% legend ('x-direction ','y-direction')
% ylim([0,1]) 
% drawnow