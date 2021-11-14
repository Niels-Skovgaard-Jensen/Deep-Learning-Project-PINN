clear all; close all;
tic;
X = 15; %Length i wavelengths
Y = 15; %Height in Wavelengths
a = 5; % Radius of dielectric in wavelengths
res = 20; % Points per wavelength
N = 150; % Number of functions to evalate fields
Nx = X*res;
Ny = Y*res;
epsilon = [1,2*ones(1,N)];
mu__0 = 1;
E__0 = 1;
eta__0 = 377;
k__0 = (2*pi)/(1);
epsilonc = 5;
mu = 1;
eta__1 = eta__0*sqrt(mu/epsilonc);
k = k__0*(sqrt(epsilonc*mu));
x = -Nx/2:Nx/2;
y = -Ny/2:Ny/2;
[xx yy] = meshgrid(x,y);
u = zeros(size(xx));
distMat = sqrt((xx/res).^2+(yy/res).^2);
u((xx.^2+yy.^2)<(a*res)^2)=-1;   % radius 100, center at the origin
phiMat = (atan2(flip(y),x'))'; %Create angle matrix

%% Calculate Incident Field
Ei = 0;
for n = 0:N
   Ei = Ei + (-1i).^(n)*E__0*epsilon(n+1).*besselj(n,k__0*distMat).*cos(n*phiMat);
end
Ei((xx.^2+yy.^2)<(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,real(Ei),'CDataMapping','scaled')

%% Calculate Field inside Dielectric
Et = 0;


for n = 0:N
   Cn = -E__0*(1j)^(-n)*eta__1*(J(n, k__0*a)*Hp(n, k__0*a) - H(n, k__0*a)*Jp(n, k__0*a))/(Jp(n, k*a)*eta__0*H(n, k__0*a) - J(n, k*a)*Hp(n, k__0*a)*eta__1);
   Et = Et + Cn.*epsilon(n+1).*besselj(n,k*distMat).*cos(n*phiMat);
end
Et(isnan(Et)) = 0;
Et((xx.^2+yy.^2)>(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,abs(Et))

%% Calculate Scattered Field
Es = 0;
for n = 0:N
   Bn = -(-1j)^(n)*E__0*(J(n, k__0*a)*Jp(n, k*a)*eta__0 - J(n, k*a)*Jp(n, k__0*a)*eta__1)/(Jp(n, k*a)*eta__0*H(n, k__0*a) - J(n, k*a)*Hp(n, k__0*a)*eta__1);
   Es = Es + Bn.*epsilon(n+1).*H(n,k__0*distMat).*cos(n*phiMat);
end
Es((xx.^2+yy.^2)<(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,abs(Es))

%% Calculate field for entire area
Etot = Ei+Et+Es;
Ei((xx.^2+yy.^2)<(a*res)^2)=0;
figure()
imagesc(x/res,y/res,abs(Etot),'CDataMapping','scaled')
title('|E| for \mu_r = 1, \epsilon_r = 5','fontsize',16);
pbaspect([1.1 1 1])
colorbar;
ab=colorbar;

ylabel(ab,'|E_0|','fontsize',13);
xlabel('x-coordinate (x/\lambda_0)','fontsize',14)
ylabel('y-coordinate (y/\lambda_0)','fontsize',14)
saveas(gca, 'Etotmu1eps5Amplitude', 'epsc');

figure()
imagesc(x/res,y/res,angle(Etot),'CDataMapping','scaled')
title('Phase of E-field for \mu_r = 1, \epsilon_r = 5','fontsize',16);
pbaspect([1.1 1 1])
colorbar;
ab=colorbar;
ylabel(ab,'arg(E)','fontsize',13);
xlabel('x-coordinate (x/\lambda_0)','fontsize',14)
ylabel('y-coordinate (y/\lambda_0)','fontsize',14)
saveas(gca, 'Etotmu1eps5Phase', 'epsc');

%% For epsilon = 5 and mu = 5
epsilonc = 5;
mu = 5;
eta__1 = eta__0*sqrt(mu/epsilonc);
k = k__0*(sqrt(epsilonc*mu));

%% Calculate Incident Field
Ei = 0;
for n = 0:N
   Ei = Ei + (1i).^(-n)*E__0*epsilon(n+1).*besselj(n,k__0*distMat).*cos(n.*phiMat);
end
Ei(X/2:end,(X/2-a):(X/2+a))
Ei((xx.^2+yy.^2)<(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,real(Ei),'CDataMapping','scaled')

%% Calculate Field inside Dielectric
Et = 0;
for n = 0:N
   Cn = -E__0*1j^(-n)*eta__1*(J(n, k__0*a)*Hp(n, k__0*a) - H(n, k__0*a)*Jp(n, k__0*a))/(Jp(n, k*a)*eta__0*H(n, k__0*a) - J(n, k*a)*Hp(n, k__0*a)*eta__1);
   Et = Et + Cn.*epsilon(n+1).*besselj(n,k*distMat).*cos(n.*phiMat);
end
Et(isnan(Et)) = 0;
Et((xx.^2+yy.^2)>(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,real(Et))

%% Calculate Scattered Field
Es = 0;
for n = 0:N
   Bn = -1j^(-n)*E__0*(J(n, k__0*a)*Jp(n, k*a)*eta__0 - J(n, k*a)*Jp(n, k__0*a)*eta__1)/(Jp(n, k*a)*eta__0*H(n, k__0*a) - J(n, k*a)*Hp(n, k__0*a)*eta__1);
   Es = Es + Bn.*epsilon(n+1).*H(n,k__0*distMat).*cos(n.*phiMat);
end
Es((xx.^2+yy.^2)<(a*res)^2)=0;
figure()
imagesc((x/res)+X/2,(y/res)+Y/2,real(Es))

%% Calculate field for entire area and Plot
Etot = Ei+Et+Es;
figure()
imagesc(x/res,y/res,abs(Etot),'CDataMapping','scaled')
title('|E| for \mu_r = 5, \epsilon_r = 5','fontsize',16);
pbaspect([1.1 1 1])
colorbar;
ab=colorbar;
ylabel(ab,'|E_0|','fontsize',13);
xlabel('x-coordinate (x/\lambda_0)','fontsize',14)
ylabel('y-coordinate (y/\lambda_0)','fontsize',14)
saveas(gca, 'Etotmu5eps5Amplitude', 'epsc');

figure()
imagesc(x/res,y/res,real(Etot),'CDataMapping','scaled')
title('\real for \mu_r = 5, \epsilon_r = 5','fontsize',16);
pbaspect([1.1 1 1])
colorbar;
ab=colorbar;
ylabel(ab,'|E_0|','fontsize',13);
xlabel('x-coordinate (x/\lambda_0)','fontsize',14)
ylabel('y-coordinate (y/\lambda_0)','fontsize',14)
saveas(gca, 'Etotmu5eps5Real', 'epsc');

figure()
imagesc(x/res,y/res,angle(Etot),'CDataMapping','scaled')
title('Phase of E-field for \mu_r = 5, \epsilon_r = 5','fontsize',16);
pbaspect([1.1 1 1])
colorbar;
ab=colorbar;
ylabel(ab,'arg(E)','fontsize',13);
xlabel('x-coordinate (x/\lambda_0)','fontsize',14)
ylabel('y-coordinate (y/\lambda_0)','fontsize',14)
saveas(gca, 'Etotmu5eps5Phase', 'epsc');
toc;
