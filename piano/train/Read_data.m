clear all
clc
for par1=2:1:5
for par2=1:1:5
letter=['A','B','E','F','G'];
Filename=[letter(par2) num2str(par1) '.wav'];
Filename_out=[letter(par2) num2str(par1) '.mat'];
[y,Fs] = audioread(Filename);

y=y(:,1);
t=0+1/Fs:1/Fs:1/Fs*size(y);
% plot(t,y,"linewidth",1.5);
% xlim([0,0.01])
% xlabel('Time')
% ylabel('Amplitude')
x=y;
% Assumes x is an N x 1 column vector
% and Fs is the sampling rate.

N = size(x,1);
dt = 1/Fs;
t = dt*(0:N-1)';
dF = Fs/N;
f = dF*(0:N/2-1)';
X = fft(x)/N;
X = X(1:N/2);
X(2:end) = 2*X(2:end);
% figure;
% plot(f,abs(X),'linewidth',1.5);
% plot(f(1:100,1),abs(X(1:100,1)),'linewidth',1.5);
% xlim([0,2000]);
% 
% xlabel('Frequency')
% ylabel('Amplitude')
X=abs(X);

X=abs(X);

[~,Index]=max(X);
o_fun=round(f(Index));

for i=1:1:8
    [row,col,v]=find(f>(i*o_fun-2)&f<(i*o_fun+2));
    omega(i)=i*o_fun;
    
end


% [~,Index]=max(X);
% basic_f=round(f(Index));
% 
% 
% TF = islocalmax(X,'MinSeparation',basic_f.*1.9);
% 
% temp=f(TF);
% omega=round(temp(1:8));

% for i=1:1:1
%     temp_in=find(f>(basic_f*i-50)&f<(basic_f*i+50));
%     [~,Index]=max(X(min(temp_in):max(temp_in)));
%     omega(i+1)=round(f(Index));
% end
% for i=1:1:8
%     [row,col,v]=find(f>(i*o_fun-2)&f<(i*o_fun+2));
%     freq_out(i)=i*o_fun;
%     amp_out(i)=max(X(row));  
% end

% figure
[s,f,t]=stft(y,Fs,'Window',	rectwin(2048*2),'FFTLength',2048*2,'FrequencyRange','onesided');
% logs=abs(s(1:300,:));
% surface=surf(t,f(1:300,1),logs,'FaceAlpha',1);
% surface.EdgeColor = 'none';
% view(0,90)
% xlabel('Time(s)')
% ylabel('Frequency(Hz)')
% colormap(jet)
%temp=[452,904,1356,1808,2260,2737];
for i=1:1:8
%   [~,Index(i)] = min(abs(f-i*o_fun));
   [~,Index(i)] = min(abs(f-omega(i)));
    amp=abs(s(Index(i),:));
    g = fittype('a*exp(b*x)');
    fit_f=fit(t,amp',g);
%     figure
%     plot(fit_f,t,amp','o');
    out=coeffvalues(fit_f);
    a_out(i)=out(1);
    b_out(i)=out(2);
end


[y,Fs] = audioread(Filename);
y=y(:,1);
t=0+1/Fs:1/Fs:1/Fs*size(y);

F = @(x,xdata)x(1).*exp(x(2).*xdata).*sin(2.*pi.*omega(1).*xdata+x(3).*pi)+x(4).*exp(x(5).*xdata).*sin(2.*pi.*omega(2).*xdata+x(6).*pi)+x(7).*exp(x(8).*xdata).*sin(2.*pi.*omega(3).*xdata+x(9).*pi)+x(10).*exp(x(11).*xdata).*sin(2.*pi.*omega(4).*xdata+x(12).*pi)+x(13).*exp(x(14).*xdata).*sin(2.*pi.*omega(5).*xdata+x(15).*pi)+x(16).*exp(x(17).*xdata).*sin(2.*pi.*omega(6).*xdata+x(18).*pi)+x(19).*exp(x(20).*xdata).*sin(2.*pi.*omega(7).*xdata+x(21).*pi);
for i=1:1:8
    x0((i-1)*3+1)=a_out(i)/173.*max(max(y));
    x0((i-1)*3+2)=b_out(i);
    x0((i-1)*3+3)=rand();
end % give the initial value of amplitude, damping coefficient and phase angle
options = optimoptions(@fminunc,'Display','iter');
[x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0,t,y',-10,10,options);
for i=1:1:8
    a(i)=x0((i-1)*3+1);
    b(i)=x0((i-1)*3+2);
    phi(i)=x0((i-1)*3+3);
end
save(Filename_out,'a','b','phi','omega');
end
end
