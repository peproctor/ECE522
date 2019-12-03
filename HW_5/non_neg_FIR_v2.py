import random as ran
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import math
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict
    

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    plt.figure(3)
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
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k

def WN_gen(length):
    ran.seed(4)
    mu = 1
    sigma = 1
    WN = np.asarray([ran.gauss(mu,sigma) for i in range(100)])

    #plt.figure(1)
    #plt.scatter(range(WN.shape[0]),WN)
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

def lp_filter(fc,fs,N,n):
    """
    Create LPF from windowed sinc
    """
    n_win = np.arange(0,N//2)
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
    wind = np.insert(wind,0,np.zeros(int((N-n_win.shape[0])/2)))
    wind = np.append(wind,np.zeros(int((N-n_win.shape[0])/2)))
    plt.figure(3)
    plt.plot(n,wind)
    plt.show()

    #Window the filter
    h_n_win = h_n*wind
    return (h_n_win / np.sum(h_n_win))

def SMA_filt(fs,N,Fc):
    if Fc == .1*np.pi:
        filt_n = 7
    elif Fc == .2*np.pi:
        filt_n = 3
    elif Fc == .3*np.pi:
        filt_n = 3
    
    b = np.ones(filt_n)
    a = np.array([filt_n] + [0]*(filt_n-1))
    hn = b/a[0]
    plt.figure(1)
    plt.stem(range(filt_n),hn)
    #plt.show()
    return b, a

def filt_resp(b,a,fs):
    plt.figure(2)
    w, mag, phase = sg.dbode((b,a,1/fs), n=1000)
    #w, h = sg.freqz(b,a,fs=fs)
    plt.subplot(2, 1, 1)
    plt.plot(w/(2*np.pi), 10**(mag/20))
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0,np.pi)
 
    plt.subplot(2, 1, 2)
    #angles = np.unwrap(np.angle(h))
    plt.plot(w/(2*np.pi), phase)
    plt.ylabel('Angle (radians)')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0,np.pi)

    #impulse_resp(b,a,fs)
    zplane(b,a)

def impulse_resp(b,a,fs):
    t,y = sg.dimpulse((b,a,1/fs))
    plt.figure(6)
    plt.step(t,np.squeeze(y))
    
def fft(sig,fs,bins=512):
    zero_pad = (8**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0])
    sig = np.append(sig, np.zeros(zero_pad))
    out = np.fft.fft(sig, n=bins)
    freq_axis_scale = (fs)/bins
    #each bin represents 10 Hz
    out_abs = np.abs(out.real)
    plt.figure(3)
    plt.plot(np.arange(3*freq_axis_scale,(fs/2),freq_axis_scale),out_abs[3:bins//2])
    #plt.xticks(np.arange(0,(fs)+1000,round(freq_axis_scale)))
    plt.xlabel('Hz')

def convolution(signal,kern,nc):
    """
    y[n] = sum x[n]*h[k-n]
    """
    sig_len = signal.shape[0] - nc
    conv_sig = np.zeros(signal.shape[0]+kern.shape[0])
    for jj in range(signal.shape[0]):
        #Beginning of conv
        if (jj==0): 
            conv_sig[jj] = np.sum(signal[0:nc]*kern[0:nc])
        
        elif ( jj > 0 ):
            for zz in range(kern.shape[0]):
                if (jj+nc) > nc:
                    conv_sig[jj] += signal[nc+jj]*kern[-zz]
                #Middle of conv
                elif jj> zz:
                    conv_sig[jj] += signal[nc+jj+zz]*kern[-zz]

    

    return conv_sig

Fs = 2*np.pi
Fc = .1*np.pi
N = 256
N_sig = 100

n = np.arange(-(N//2),(N//2)+1)
noise_sig = WN_gen(N_sig)
#n = np.arange(0,(N))
#h_n = lp_filter(Fc,Fs,N,n)
b, a = SMA_filt(Fs,N,Fc)

sig_conv = convolution(noise_sig,b/a[0], (b.shape[0]//2))

filt_resp(b,a,Fs)

plt.figure(4)
plt.plot(range(N_sig),noise_sig)
plt.plot(range(N_sig),sg.lfilter(b,a,noise_sig),c='red')
#plt.show()

fft(noise_sig,Fs)
plt.show()



print('None')