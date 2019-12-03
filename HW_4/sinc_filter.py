import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

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
    n_pos = np.arange(int(N/2))
    fc_scaled = fc/fs

    #(n-(n-1)/2) shifts the sinc to the midpoint
    h_n = np.sinc(2*fc_scaled*(n-(N-1)/2)) 
    
    #Form the blackman and shift it to the same time as the signal
    wind = blackman_win(n_pos,N_filt)
    wind = np.insert(wind,0,np.zeros(int((N-N_filt)/2)))
    wind = np.append(wind,np.zeros(int((N-N_filt)/2)))

    #Window the filter
    h_n_win = h_n*wind
    return (h_n_win / np.sum(h_n_win))

def hp_filter(fc,fs,N,n):
    """
    Create LPF then negate and add 1 to the center
    """
    n_pos = np.arange(int(N/2))
    fc_scaled = fc/fs

    #(n-(n-1)/2) shifts the sinc to the midpoint
    h_n = np.sinc(2*fc_scaled*(n-(N-1)/2)) 
    
    #Form the blackman and shift it to the same time as the signal
    wind = blackman_win(n_pos,N_filt)
    wind = np.insert(wind,0,np.zeros(int((N-N_filt)/2)))
    wind = np.append(wind,np.zeros(int((N-N_filt)/2)))
    h_n_win = h_n*wind
    h_n_win = (h_n_win / np.sum(h_n_win))

    h_n_win = -h_n_win
    h_n_win[(N-1)//2] += 1

    return h_n_win

fs = 128
fc_low = 12
fc_high = 20
#fc_sca = fc_low/fs
N = 512
N_filt = 256
n = np.arange(N)
n_pos=np.arange((int(N/2)))

x_n = np.cos(2*np.pi*4*n_pos) + np.cos(2*np.pi*36*n_pos)

win_lpf = lp_filter(fc_low,fs,N,n)
#win_hpf = hp_filter(fc_high,fs,N,n)

lp_freq, scale_win = freq_resp(win_lpf,fs)
#hp_freq, scale_win = freq_resp(win_hpf,fs)

#FFT the input signal
x_n_freq, scale_x_n = freq_resp(x_n,fs)
y_n_filt = np.convolve(win_lpf,x_n)


plt.figure(1)
plt.plot(n,win_lpf)
plt.grid()
plt.figure(2)
plt.title("DFT impulse response")
plt.plot(np.arange(0,(fs/2),scale_win),lp_freq[0:int(fs/(2*scale_win))])
plt.xticks(np.arange(0,(fs/2),10))
plt.xlabel('Hz')

"""
plt.figure(3)
plt.title("DFT impulse response")
plt.plot(np.arange(0,(fs/2),scale_win),hp_freq[0:int(fs/(2*scale_win))])
plt.xticks(np.arange(0,(fs/2),10))
plt.xlabel('Hz')
"""

plt.figure(4)
plt.title("Input signal DFT")
plt.plot(np.arange(1,(fs/2),scale_x_n),x_n_freq[4:int(fs/(2*scale_x_n))])
plt.xticks(np.arange(0,(fs/2),10))
plt.grid()
plt.show()