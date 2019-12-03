from scipy.io import wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import math as math


def peaks_data(file_p):
    fs, data_extract = wavfile.read(file_p)
    data_extract = data_extract[0:(int(data_extract.shape[0]))]
    peaks_extract, _ = sig.find_peaks(data_extract,height=1400)
    peaks_plot = np.empty(data_extract.shape[0],dtype=float)
    peaks_plot.fill(np.nan)
    peaks_plot[peaks_extract] = data_extract[peaks_extract]
    return fs,data_extract, peaks_extract, peaks_plot

def butter_hp(cutoff, fs, bs_type, order=5):
    #norm_cutoff = 2*cutoff / fs
    b,a = sig.butter(order,cutoff,bs_type, fs= fs)
    #w, h = sig.freqz(b, a)   
    #plt.semilogx(w, 20 * np.log10(abs(h)))
    #plt.show()
    return sig.filtfilt(b,a,data)

def iir_filt(order, freq,b_type,f_type, fs, data):
    #norm_cutoff = (2*w0) / fs #w0 / nyq
    b,a = sig.iirfilter(order, freq,btype=b_type,ftype=f_type, fs=fs)
    w, h = sig.freqz(b, a)   
    plt.figure(10)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.show()
    return sig.filtfilt(b,a,data)
    


def fft(sig,fs,bins=512):
    zero_pad = 2**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0]
    sig = np.append(sig, np.zeros(zero_pad))
    out = np.fft.fft(sig, n=bins)
    freq_axis_scale = (fs)/bins
    #each bin represents 10 Hz
    out_abs = np.abs(out.real)
    plt.figure(3)
    plt.title("DFT of High Pass Filtered Signal")
    plt.plot(np.arange(0,(fs),freq_axis_scale),out_abs)
    plt.xticks(np.arange(0,(fs)+1000,round(freq_axis_scale)*100))
    plt.xlabel('Hz')

    #plt.figure(4)
    #freq = np.fft.fftfreq(n=bins, d=1/fs)
    #plt.plot(freq, np.abs(out.real))
    #plt.xlim(-10,5000)
    plt.show()
    
    

file_p = '/Users/pproctor/Documents/PSU/ECE522/HW_2/Henderson2.wav'
fs, data, peaks, peaks_plot = peaks_data(file_p)
binwidth = 300

#Highpass filter
filt_dc = 1
filt_dc_bs = 100 #80 is good
btype = 'highpass'
order = 6
#Notch filter

fft(data,fs)
#data_filt = DC_notch_filt(filt_dc, Q_dc, fs, data)
data_filt = butter_hp(filt_dc_bs,fs,btype,order)
f_peaks, _ = sig.find_peaks(data_filt, height=1400)
f_peaks_plot = np.empty(data_filt.shape[0],dtype=float)
f_peaks_plot.fill(np.nan)
f_peaks_plot[f_peaks] = data_filt[f_peaks]
fft(data_filt, fs)

plt.figure(1)
plt.title("Raw data")
plt.plot(range(data.shape[0]), data)
plt.scatter(range(peaks_plot.shape[0]),peaks_plot, c='red', s=8)
#plt.plot(range(data.shape[0]), data_filt, c='black')

plt.figure(2)
plt.title("Filtered data peaks")
plt.plot(range(data_filt.shape[0]), data_filt)
plt.scatter(range(f_peaks_plot.shape[0]),f_peaks_plot, c='red', s=8)

plt.figure(4)
plt.title("Raw w/o threshold data")
min_bin = np.min(data[peaks])
max_bin = np.max(data[peaks])
max_bin_2 = 32444
min_bin_2 = -21231
print("Type min_bin:",type(min_bin),"max_bin:",type(max_bin))
print("min_bin:", min_bin,"max_bin:",max_bin)
print("Type data", type(data[peaks]))
plt.hist(data[peaks], bins=np.arange(min_bin_2,max_bin_2, binwidth))

plt.figure(5)
plt.title("Filtered data")
plt.hist(data_filt[f_peaks], bins=np.arange(min_bin_2,max_bin_2, binwidth))

"""
order_2 = 4
b_type = 'bandstop'
f_type = 'butter'
filt_freq = (660,670)
data_filt_n = iir_filt(order_2,filt_freq,b_type,f_type,fs,data_filt)
fft(data_filt_n,fs)
plt.figure(6)
plt.title("Notch Filtered data peaks")
plt.plot(range(data_filt_n.shape[0]), data_filt_n)
"""

order_2 = 2
b_type = 'highpass'
f_type = 'butterworth'
filt_freq = 100
data_filt_lp = iir_filt(order_2,filt_freq,b_type,f_type,fs,data)
f_peaks_lp, _ = sig.find_peaks(data_filt_lp, height=1400)
f_peaks_plot_lp = np.empty(data_filt_lp.shape[0],dtype=float)
f_peaks_plot_lp.fill(np.nan)
f_peaks_plot_lp[f_peaks_lp] = data_filt_lp[f_peaks_lp]

plt.figure(6)
plt.title("LP Filtered data")
plt.hist(data_filt_lp[f_peaks], bins=np.arange(min_bin_2,max_bin_2, binwidth))

plt.figure(7)
plt.title("Low Pass Filtered data peaks")
plt.plot(range(data_filt_lp.shape[0]), data_filt_lp)
plt.scatter(range(f_peaks_plot_lp.shape[0]),f_peaks_plot_lp, c='red', s=8)
plt.show()
fft(data_filt_lp,fs)

#plt.show()
print("None")