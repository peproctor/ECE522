from scipy.io import wavfile
import scipy.signal as sig_p
import matplotlib.pyplot as plt
import numpy as np
import math as math
import os

def fft(sig,fs,bins=128):
    zero_pad = 2**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0]
    sig = np.append(sig, np.zeros(zero_pad))
    out = np.fft.fft(sig, n=bins)
    freq_axis_scale = (fs)/bins
    #each bin represents 10 Hz
    out_abs = np.abs(out.real)
    #plt.figure(3)
    #plt.title("DFT of Signal")
    #plt.plot(np.arange(0,(fs/2),freq_axis_scale),out_abs[0:int(fs/2)])
    #plt.xticks(np.arange(0,(fs/2),freq_axis_scale))
    #plt.xlabel('Hz')

def SMA_filt(signal,size,fs, filtfilt=True):
    # SMA coefficients
    b = np.ones(size)
    a = np.array([size] + [0]*(size-1))
    #bode(b,a,fs)
    if filtfilt is True:
        return sig_p.filtfilt(b,a,signal)
    else:
        return sig_p.lfilter(b,a,signal)

def low_pass(signal, cutoff,fs, filtfilt=True):
    b,a = sig_p.iirfilter(4, cutoff, btype= 'lowpass',ftype='butter',fs=fs)
    if filtfilt is True:
        return sig_p.filtfilt(b,a,signal)
    else:
        return sig_p.lfilter(b,a,signal)

def bode(num,den,fs):
    
    #w,mag,phase = sig_p.dbode((b,a,1/fs),n=1000)
    w,mag,phase = sig_p.dbode((num,den,1/fs), n=1000)
    plt.figure(5)
    plt.title('Mag')
    plt.plot((w/(2*math.pi)),mag)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude [dB]')

    plt.figure(6)
    plt.title('Phase')
    plt.plot((w/(2*math.pi)),phase)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Phase [deg]')
    #plt.show()

def maxima_det(signal,thresh):
        max_list = [0]
        max_ls = 0
        if thresh >= 0:
                for jj in range(signal.shape[0]):
                        if signal[jj] > thresh and (signal[jj]-signal[jj-1] > -.002 and signal[jj]-signal[jj-1] < 0):
                                max_list.append(jj)
                                max_ls += 1
        else:
                for jj in range(signal.shape[0]):
                        if signal[jj] < thresh and (signal[jj]-signal[jj-1] > 0 and signal[jj]-signal[jj-1] < .07):
                                max_list.append(jj)
        
        return max_list

def interval_calc(peaks_ls,data):
        int_ls = []
        time_ls = []
        for jj in range(peaks_ls.shape[0]):
                if jj < (peaks_ls.shape[0]-1):
                        time = data[peaks_ls[jj+1]]-data[peaks_ls[jj]] 
                        interval = (data[peaks_ls[jj]] + data[peaks_ls[jj+1]])/2
                        time_ls.append(time)
                        int_ls.append(interval)
        return int_ls, time_ls

fs = 128 #Hz
T = 1/fs 
window = 5

#data = np.ndarray(shape=(5,2000), dtype=float)
#data.fill(np.nan)
data = {}
sig = 2

file_p = '/Users/pproctor/Documents/PSU/ECE522/HW_3'
file_ls = os.listdir(file_p)
zz = 0
for da_file in file_ls:
    if (da_file[-4:] == '.txt'):
        data[zz] = np.loadtxt(os.path.join(file_p,da_file))
        zz+=1

#fft(data[sig],fs)

#SMA filt
data_filt_1 = SMA_filt(data[sig],window,fs,False)
data_filt_2 = SMA_filt(data[sig],window,fs,True)
#LP butter
data_filt_3 = SMA_filt(data[sig],12,fs,False)
data_filt_4 = SMA_filt(data[sig],12,fs,True)

#Peak detection
x= np.arange(0,data[sig].shape[0])*T
peaks_pos, _p = sig_p.find_peaks(data_filt_2,height=(.15,None),distance=20)
peaks_neg, _n = sig_p.find_peaks(-data_filt_2,height=(.15,None),distance=20)

peaks_pos_uf, _p = sig_p.find_peaks(data[sig],height=(.15,None),distance=20)
peaks_neg_uf, _n = sig_p.find_peaks(-data[sig],height=(.15,None),distance=20)

#Calc intervals
peaks_max_int_uf, time_max_int_uf = interval_calc(peaks_pos_uf,x)
peaks_min_int_uf, time_min_int_uf = interval_calc(peaks_neg_uf,x)

peaks_max_int, time_max_int = interval_calc(peaks_neg,x)
peaks_min_int, time_min_int = interval_calc(peaks_neg,x)

plt.figure(1)
plt.plot(x,data[sig])
#plt.plot(x,data_filt_1, label='Filt')
plt.plot(x,data_filt_2, label= 'Filt-Filt')
#plt.scatter(peaks_pos*T,data[sig][peaks_pos],c='black',s=15)
plt.scatter(peaks_neg*T,data[sig][peaks_neg],c='black',s=15)
plt.scatter(peaks_pos_uf*T,data[sig][peaks_pos_uf],c='purple',s=20, marker='*')
plt.scatter(peaks_neg_uf*T,data[sig][peaks_neg_uf],c='purple',s=20, marker='*')
plt.xlabel("Time [s]")
plt.ylabel("Velocity [M/s]")
plt.legend()


plt.figure(3)
plt.title("Max peaks")
plt.scatter(peaks_max_int_uf,time_max_int_uf, marker='*', c='purple')
plt.scatter(peaks_max_int,time_max_int)
plt.ylim(.2,.6)
plt.xlabel("Time [s]")
plt.ylabel("Interval Len. [s]")
plt.figure(4)
plt.title("Min peaks")
plt.scatter(peaks_min_int_uf,time_min_int_uf, marker='*',c='purple')
plt.scatter(peaks_min_int,time_min_int)
plt.xlabel("Time [s]")
plt.ylabel("Interval Len. [s]")
plt.ylim(.2,.6)


plt.show()

print("none")
