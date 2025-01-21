close all; clear; clc;

%% Problem 1

% i) Plot p(x) via coefficients 
x = 1.92:0.001:2.08;

p = x.^9 -18*x.^8 +144*x.^7 -672*x.^6 +2016*x.^5 -4032*x.^4 +5376*x.^3 -4608*x.^2 +2304*x-512;

figure()
plot(x,p,'Linewidth',1);
grid minor
xlabel('x')
ylabel('p(x)')
title('p(x) Evaluated by Coefficients')

% ii) Plot by (x-2)^2

p = (x-2).^9;

figure()
plot(x,p,'Linewidth',1);
grid minor
xlabel('x')
ylabel('p(x)')
title('p(x) Evaluated by as (x-2)^9')