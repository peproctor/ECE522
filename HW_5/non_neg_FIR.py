import random as ran
import numpy as np
import matplotlib.pyplot as plt
import math

def WN_gen(length):
    #ran.seed(1)
    WN = np.asarray([ran.gauss(1,1) for i in range(100)])
    plt.figure(1)
    plt.scatter(range(WN.shape[0]),WN)
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
    plt.show()
    wind = np.insert(wind,0,np.zeros(int((N-n_win.shape[0])/2)))
    wind = np.append(wind,np.zeros(int((N-n_win.shape[0])/2)))
    plt.figure(3)
    plt.plot(n,wind)
    plt.show()

    #Window the filter
    h_n_win = h_n*wind
    return (h_n_win / np.sum(h_n_win))


Fs = 128
Fc = 32
N = 256
n = np.arange(-(N//2),(N//2)+1)
#n = np.arange(0,(N))
h_n = lp_filter(Fc,Fs,N,n)



print('None')