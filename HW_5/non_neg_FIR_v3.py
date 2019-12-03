import random as ran
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import math
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict
    
"""
Todo:
*One more low pass filter hamming window
*Do highpass filters
"""


def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    plt.figure(3)
    plt.title("WMA PZ plot")
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        #plt.savefig(file_zp)
        print("NonE")
    else:
        plt.savefig(filename)
    

    return z, p, k

def WN_gen(length):
    ran.seed(4)
    mu = 1
    sigma = 1
    WN = np.asarray([ran.gauss(mu,sigma) for i in range(100)])

    plt.figure(1)
    plt.title("Raw White Noise Signal")
    plt.plot(np.arange(WN.shape[0])*(1/Fs),WN)
    plt.xlabel("Time [s]")
    #plt.show()
    #plt.savefig("/Users/pproctor/Documents/PSU/ECE522/HW_5/raw_white.png")
    #fft(WN,Fs)
    print("WN mean:",np.mean(WN))
    print("WN var:", np.var(WN))

    return WN

def blackman_win(n_pts,N_pts):
    term_1 = .5*np.cos((2*np.pi*n_pts/(N_pts-1)))
    term_2 = .08*np.cos((4*np.pi*n_pts/ (N_pts-1)))
    return (.42 - term_1 + term_2)

def freq_resp(sig,fs, bins=512):
    zero_pad = 2**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0]
    sig = np.append(sig, np.zeros(zero_pad))
    freq_axis_scale = (fs)/bins
    H_n = np.fft.fft(sig,n=bins)
    out_abs = np.abs(H_n)
    return out_abs, freq_axis_scale

def find_fc(fs):
    cutoff_ls ={}
    rang = np.arange(4,100)
    for n in rang:
        b,a = blackman_filt(n,1)
        print(n)
        wc_freq = filt_resp_iter(b,a,fs)
        cutoff_ls[n] = wc_freq

    plt.figure(9)
    plt.title("Blackman Filter length vs. cutoff freq.")
    plt.scatter(cutoff_ls.keys(),cutoff_ls.values())
    plt.xlabel("Length [samples]")
    plt.ylabel("Cutoff Freq. [Hz]")
    
    #plt.savefig("/Users/pproctor/Documents/PSU/ECE522/HW_5/freq_sam_black.png")
    plt.show()
    

def lp_filter(fc,fs,N,n):
    """
    Create LPF from windowed sinc
    """
    n_win = np.arange(0,N//2+1)
    fc_scaled = fc/fs

    #(n-(n-1)/2) shifts the sinc to the midpoint
    h_n = np.sinc(fc_scaled*n) 
    plt.figure(1)
    plt.plot(n,h_n)
    #Form the blackman and shift it to the same time as the signal
    wind = blackman_win(n_win, n_win.shape[0])
    n_win = n_win - n_win.shape[0]//2
    plt.figure(2)
    plt.plot(n_win,wind)
    #plt.show()
    wind = np.insert(wind,0,np.zeros(int((N-n_win.shape[0])/2 + 1)))
    wind = np.append(wind,np.zeros(int((N-n_win.shape[0])/2 + 1)))
    #plt.figure(3)
    #plt.plot(n,wind)
    #plt.show()

    #Window the filter
    h_n_win = h_n*wind
    return np.abs((h_n_win / np.sum(h_n_win)))

def SMA_filt(fs,N,Fc):
    
    if Fc == .1*np.pi:
        filt_n = 7
    elif Fc == .2*np.pi:
        filt_n = 5
    elif Fc == .3*np.pi:
        filt_n = 3
    
    
    b = np.ones(filt_n)
    a = np.array([filt_n] + [0]*(filt_n-1))
    b = (b/a[0])#*-1
    #b[b.shape[0]//2] += 1
    plt.figure(13)
    plt.title("SMA Impulse Resp.")
    plt.xlabel("Samples")
    plt.stem(np.arange((-filt_n//2)+1,(filt_n//2)+1,1),b)
    #plt.savefig(file_impulse)
    plt.show()
    return b, a

def EMA_filt(fs, N,Fc):
    
    if Fc == .1*np.pi:
        filt_n = 9
        alpha = .98
    elif Fc == .2*np.pi:
        filt_n = 5
        alpha = .98
    elif Fc == .3*np.pi:
        filt_n = 3
        alpha = .98
    

    #filt_n = N
    #alpha = .95
    
    b_ema = np.array((alpha**(np.arange(0,(filt_n//2)+1))))
    b_ema = np.concatenate((b_ema[:0:-1],b_ema))
    a_ema = np.array([filt_n] + [0]*(filt_n-1))
    b_ema = (b_ema/a_ema[0])#*-1
    #b_ema[b_ema.shape[0]//2] += 1
    #hn = hn / np.amax(hn)
    
    plt.figure(12)
    plt.title("WMA Impulse Resp.")
    plt.xlabel("Samples")
    plt.stem(np.arange((-filt_n//2)+1,(filt_n//2)+1,1),b_ema)
    #plt.savefig(file_impulse)
    plt.show()
    return b_ema, a_ema

def blackman_filt(Fc):
    
    if Fc == .1*np.pi:
        N = 17
    elif Fc == .2*np.pi:
        N = 9
    elif Fc == .3*np.pi:
        N =7
    
    #N=n
    n_win = np.arange(0,N)
    b_win = blackman_win(n_win,n_win.shape[0])
    b_win = (b_win/np.sum(b_win))#*-1
    #b_win[b_win.shape[0]//2] += 1
    a_win = np.array([1]+([0]*(b_win.shape[0]-1)))
    plt.figure(20)
    plt.title("Blackman Impulse Resp.")
    plt.xlabel("Samples")
    plt.plot(np.arange((-N//2)+1,(N//2)+1,1),b_win)
    #plt.savefig(file_impulse)
    plt.show()
    
    return b_win, a_win

def tri_filt(n,Fs, Fc):
    """
    if Fc == .1*np.pi:
        m = .15
    elif Fc == .2*np.pi:
        m = .27
    elif Fc == .3*np.pi:
        m = .38
    """
    if Fc == .1*np.pi:
        z = 6
    elif Fc == .2*np.pi:
        z = 3
    elif Fc == .3*np.pi:
        z = 2
    
    m = 1
    #m = .15
    #z = 
    n_set = np.arange(0,(z/m)+1)
    b = z - m*n_set 
    b = np.concatenate((b[:0:-1],b))
    a = np.array([1] + [0]*(b.shape[0]-1))
    b = b/np.sum(b)#*-1
    #b[b.shape[0]//2] += 1

    plt.figure()
    plt.title("Tri Impulse Resp.")
    plt.plot(np.arange(-(b.shape[0]//2), (b.shape[0]//2)+1), b)
    plt.xlabel("Samples")
    #plt.savefig(file_impulse)
    plt.show()
    
    return b,a

def hamming_filt(n, Fc):
    
    if Fc == .1*np.pi:
        N = 19
    elif Fc == .2*np.pi:
        N = 7
    elif Fc == .3*np.pi:
        N = 5
    
    #N = n
    n_win = np.arange(0,N)
    b_win = .54 - .46*np.cos((2*np.pi*n_win/(n_win.shape[0]-1)))
    a_win = np.array([1]+([0]*(b_win.shape[0]-1)))
    
    b_win = (b_win/np.sum(b_win))#*-1
    #b_win[b_win.shape[0]//2] += 1
    plt.figure(3)
    plt.title("Hamming Impulse Resp.")
    plt.xlabel("Samples")
    plt.plot(np.arange((-N//2)+1,(N//2)+1,1),b_win)
    #plt.savefig(file_impulse)
    plt.show()

    return b_win, a_win


def filt_resp_iter(b,a,fs):
    plt.figure(2)
    w, mag, phase = sg.dbode((b,a,1/fs), n=1000)
    wc = np.where((10**(mag/20)) >= .707)
    print("Mag cutoff:", 10**(mag[wc[0][-1]]/20))
    return (w[wc[0][-1]]/(2*np.pi))

def filt_resp(b,a,fs):   
    plt.figure(2)
    w, mag, phase = sg.dbode((b,a,1/fs), n=1000)
    plt.subplot(2, 1, 1)
    plt.title("WMA Mag. and Phase Response")
    plt.plot(w/(2*np.pi), 10**(mag/20))
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0,np.pi)
    
    plt.subplot(2, 1, 2)
    plt.plot(w/(2*np.pi), np.unwrap(phase))
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0,np.pi)
    #plt.savefig(file_mag)

    zplane(b,a)
    group_delay(w,phase,b,a)
    

    
def fft(sig,fs,bins=512):
    zero_pad = (8**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0])
    sig = np.append(sig, np.zeros(zero_pad))
    out = np.fft.fft(sig, n=bins)
    freq_axis_scale = (fs)/bins
    #each bin represents 10 Hz
    out_abs = np.abs(out.real)
    plt.figure(10)
    plt.title("WMA HP Filtered Power Spectral Density")
    plt.plot(np.arange(8*freq_axis_scale,(fs/2),freq_axis_scale),out_abs[8:bins//2])
    #plt.xticks(np.arange(0,(fs)+1000,round(freq_axis_scale)))
    plt.xlabel('Hz')
    plt.ylabel("|H(e^jw)|")
    #plt.show()
    #file_psd = "/Users/pproctor/Documents/PSU/ECE522/HW_5/ham_hp_filt_psd.png"
    #plt.savefig(file_psd)
   

def convolution(signal,kern,nc):
    """
    y[n] = sum x[n]*h[k-n]
    """
    og_sig_len = signal.shape[0]
    conv_sig = np.zeros(signal.shape[0]+kern.shape[0])
    init = True
    if init is True:
        mu = .5
        sigma = 1
        WN_init = np.asarray([ran.gauss(mu,sigma) for i in range(nc-1)])
        WN_end = np.asarray([ran.gauss(mu,sigma) for i in range(2*kern.shape[0]-1)])
        signal = np.concatenate((WN_init,signal))
        signal = np.concatenate((signal,WN_end))

    for jj in range(conv_sig.shape[0]):
        #Beginning of conv
        conv_sig[jj] = np.sum(signal[jj:(jj+kern.shape[0])]*kern)

    return conv_sig

def group_delay(w,phase,b,a):
    w, gd = sg.group_delay((b, a),w=w)
    plt.figure(6)
    plt.title('WMA filter group delay')
    plt.plot(w[1:990]/(2*np.pi),np.round(gd[1:990]))
    plt.ylabel('Group delay [samples]')
    plt.xlabel('Frequency [Hz]')
    plt.savefig(file_gd)
    #plt.show()

    #plt.figure(6)
    #plt.title('Blackman filter group delay')
    #plt.plot(w[:999]/(2*np.pi),np.diff(np.unwrap(phase)))
    #plt.ylabel('Group delay [samples]')
    #plt.xlabel('Frequency [Hz]')
    
    
    

Fs = 2*np.pi
Fc = .1*np.pi
N = 256
N_sig = 100

n = np.arange(-(N//2),(N//2)+1)
noise_sig = WN_gen(N_sig)
#n = np.arange(0,(N))
#h_n = lp_filter(Fc,Fs,N,n)

#file_impulse = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_impulse_fc_hp.png"
#file_mag = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_phase_mag_fc_hp.png"
file_gd = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_group_delay_fc_3.png"
#file_zp = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_zplane_fc_hp.png"
#file_conv = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_conv_fc_hp.png"
#file_psd = "/Users/pproctor/Documents/PSU/ECE522/HW_5/wma_psd.png"


#Hamming
#find_fc(Fs)
#find_fc(Fs)
"""
b_ham, a_ham = hamming_filt(31,1*Fc)
sig_conv = convolution(noise_sig,b_ham, (b_ham.shape[0]//2))
fft(sig_conv, Fs)
filt_resp(b_ham,a_ham,Fs)
plt.figure(4)
plt.title("Raw signal and SMA filtered")
plt.plot(np.arange(N_sig)*(1/Fs),noise_sig,label='Raw Signal')
plt.plot(np.arange(sig_conv[:100].shape[0])*(1/Fs),sig_conv[:100], c='orange', label='Filtered')
plt.xlabel("Time [s]")
plt.legend()
#plt.savefig(file_conv)
plt.show()
"""
#SMA
"""
b_sma, a_sma = SMA_filt(Fs,N,3*Fc) 
sig_conv = convolution(noise_sig,b_sma, (b_sma.shape[0]//2))
fft(sig_conv, Fs)
filt_resp(b_sma,a_sma,Fs)
plt.figure(4)
plt.title("Raw signal and SMA filtered")
plt.plot(np.arange(N_sig)*(1/Fs),noise_sig,label='Raw Signal')
plt.plot(np.arange(sig_conv[:100].shape[0])*(1/Fs),sig_conv[:100], c='orange', label='Filtered')
plt.xlabel("Time [s]")
plt.legend()
#plt.savefig(file_conv)
plt.show()
"""
#EMA_filt

b_ema, a_ema = EMA_filt(Fs, N, 3*Fc)
filt_resp(b_ema, a_ema, Fs)
sig_conv = convolution(noise_sig,b_ema, (b_ema.shape[0]//2))
fft(sig_conv, Fs)

plt.figure(4)
plt.title("Raw signal and WMA filtered")
plt.plot(np.arange(N_sig)*(1/Fs),noise_sig,label='Raw Signal')
plt.plot(np.arange(sig_conv[:100].shape[0])*(1/Fs),sig_conv[:100], c='orange', label='Filtered')
plt.xlabel("Time [s]")
#plt.savefig(file_conv)
plt.legend()



#Blackman window_filt
"""
b_win, a_win = blackman_filt(3*Fc)
filt_resp(b_win,a_win, Fs)
sig_conv = convolution(noise_sig,b_win, (b_win.shape[0]//2))
fft(sig_conv, Fs)

plt.figure(4)
plt.title("Raw signal and Blackman filtered")
plt.plot(np.arange(N_sig)*(1/Fs),noise_sig,label='Raw Signal')
#plt.plot(range(N_sig),sg.lfilter(b_tri,a_tri,noise_sig),c='red')
plt.plot(np.arange(sig_conv[:100].shape[0])*(1/Fs),sig_conv[:100], c='orange', label='Filtered')
plt.xlabel("Time [s]")
plt.legend()
#plt.savefig(file_conv)
plt.show()
"""


#Tri_filt
"""
b_tri, a_tri = tri_filt(30,Fs,3*Fc)
filt_resp(b_tri,a_tri,Fs)
sig_conv = convolution(noise_sig,b_tri, (b_tri.shape[0]//2))
fft(sig_conv, Fs)

plt.figure(4)
plt.title("Raw signal and Tri filtered")
plt.plot(np.arange(N_sig)*(1/Fs),noise_sig,label='Raw Signal')
#plt.plot(range(N_sig),sg.lfilter(b_tri,a_tri,noise_sig),c='red')
plt.plot(np.arange(sig_conv[:100].shape[0])*(1/Fs),sig_conv[:100], c='orange', label='Filtered')
plt.xlabel("Time [s]")
plt.legend()
#plt.savefig(file_conv)
plt.show()
"""

#LP_filt
#b_lpf = lp_filter(Fc,Fs,N,n)
#a_lpf = np.array([1]+([0]*b_lpf.shape[0]))
#filt_resp(b_lpf,a_lpf,Fs)

#fft(noise_sig,Fs)
plt.show()



print('None')