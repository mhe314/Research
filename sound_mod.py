import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from matplotlib.pyplot import xlim
from scipy.io import wavfile,savemat
from scipy.fftpack import fft, fftfreq
from scipy.signal import argrelextrema,find_peaks,stft
from scipy.optimize import curve_fit

# Define fit funtion:optimze all the features
# def func1(x, a, b):
# return a * np.exp(b * x)
# def func2(t,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7,omega):
# f=a0*np.exp(b0 * t)*np.sin(omega[0]*2*np.pi*t+c0)+a1*np.exp(b1 * t)*np.sin(omega[1]*2*np.pi*t+c1)+a2*np.exp(b2 * t)*np.sin(omega[2]*2*np.pi*t+c2)+a3*np.exp(b3 * t)*np.sin(omega[3]*2*np.pi*t+c3)+a4*np.exp(b4 * t)*np.sin(omega[4]*2*np.pi*t+c4)+a5*np.exp(b5 * t)*np.sin(omega[5]*2*np.pi*t+c5)+a6*np.exp(b6 * t)*np.sin(omega[6]*2*np.pi*t+c6)+a7*np.exp(b7 * t)*np.sin(omega[7]*2*np.pi*t+c7)
# return f
# def func3(t,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7,):
# return func2(t,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7,omega)

#Define fit funtion: optimize phase angles
def func1(x, a, b):
    return a * np.exp(b * x)
def func2(t,c0,c1,c2,c3,c4,c5,c6,c7,a,b,omega):
    f=a[0]*np.exp(b[0]* t)*np.sin(omega[0]*2*np.pi*t+c0)+a[1]*np.exp(b[1] * t)*np.sin(omega[1]*2*np.pi*t+c1)+a[2]*np.exp(b[2] * t)*np.sin(omega[2]*2*np.pi*t+c2)+a[3]*np.exp(b[3] * t)*np.sin(omega[3]*2*np.pi*t+c3)+a[4]*np.exp(b[4] * t)*np.sin(omega[4]*2*np.pi*t+c4)+a[5]*np.exp(b[5] * t)*np.sin(omega[5]*2*np.pi*t+c5)+a[6]*np.exp(b[6] * t)*np.sin(omega[6]*2*np.pi*t+c6)+a[7]*np.exp(b[7] * t)*np.sin(omega[7]*2*np.pi*t+c7)
    return f
def func3(t,c0,c1,c2,c3,c4,c5,c6,c7):
    return func2(t,c0,c1,c2,c3,c4,c5,c6,c7,a,b,omega)

## Read sound file
Fs, sound_data = wavfile.read('A4.wav')
y_data=sound_data[:,0]
t_data=np.linspace(0+1/Fs,1/Fs*y_data.size,y_data.size)
y=y_data
t=t_data

## FFT
N = y.size
dt = 1/Fs
t = dt*np.linspace(0,N-1,num=N)
dF = Fs/N
f = dF*np.linspace(0,N/2-1,num=round(N/2))
X = fft(y)/N;
X = X[0:round(N/2)];
X[1:] = 2*X[1:];
X=abs(X);

## Plot FFT
plt.plot(f, X)
xlim(0,3000);   # Define x axix limitation in the figure
plt.grid()
plt.show()

## Find foundamental frequencies
Index=np.argmax(X)
basic_f=round(f[Index])
TF=find_peaks(X,height=None, threshold=None, distance=round(Index))
TF=TF[0]
temp1=f[TF]
omega=temp1[0:8]
print(omega)

#stft
f,t,s=stft(y,Fs,window='boxcar',nperseg=2048*2,noverlap=None, nfft=None,detrend=False, return_onesided=True,);
f_plot=f[0:400]
s_plot=np.log(np.abs(s[1:400,]))
plt.pcolormesh(t,f_plot,s_plot)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#Find initial guess of a and b
a=np.zeros(8)
b=np.zeros(8)
for i in range(0, 7):
    Index=np.argmin(np.abs(f-omega[i]))
    amp=np.abs(s[Index,])
    popt, pcov = curve_fit(func1, t, amp)
    a[i]=popt[0]
    b[i]=popt[1]
c_ini=np.random.random(8)
#p=np.concatenate((a, b,c_ini)) uncommend if you want to optimize all the features
p=c_ini

popt, pcov = curve_fit(func3, t_data, y_data,p)

phi=popt
#for i in range(0, 7): #uncommend if you want to optimize all the features
    #a_out[i]=popt[i]
    #b_out[i]=popt[i+8]
    #c_out[i]=popt[i+16]

mdic={"a":a,"b":b,"phi":phi,"omega":omega}
savemat('A4.mat',mdic)