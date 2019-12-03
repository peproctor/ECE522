import numpy as np
import math
import random as ran
import matplotlib.pyplot as plt


def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    plt.figure(3)
    plt.title("WMA PZ plot")
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = plt.patches.Circle((0,0), radius=1, fill=False,
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

def WN_gen(length,Fs):
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

def dft(signal,fs,bins):
    zero_pad = (2**math.ceil(math.log(signal.shape[0],2)) - signal.shape[0])
    signal = np.append(signal, np.zeros(zero_pad))
    freq_axis_scale = (fs)/bins
    sig_coeff = np.zeros(signal.shape[0])

    for zz in range(fs//2):
        for nn in range(signal.shape[0]):
            cm_co = math.e**(-2j*math.pi*zz*nn/(signal.shape[0]))
            sig_coeff[zz] += signal[nn]*cm_co

    plt.figure(1)
    plt.plot(np.arange(0,fs,fs/sig_coeff.shape[0]),sig_coeff)
    plt.show()

def main():
    f = 20 #Hz
    fs = 100 #Hz
    bins = 512
    n = np.linspace(0,.5,1000)
    input_sig = np.sin(2*np.pi*n*f) + np.cos(2*np.pi*n*10)
    plt.figure(2)
    plt.plot(n,input_sig)
    plt.show()
    dft(input_sig,fs,bins)

if __name__ == "__main__":
    main()