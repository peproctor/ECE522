import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy.io
import signal_lib
import scipy.signal as sg
import math

def load_mat(file_1,file_2,type_s='default'):
    ecg = scipy.io.loadmat(file_1)
    x = scipy.io.loadmat(file_2)
    if type_s is 'default':
        x_1 = x['x'][0][50:140]
        ecg_1 = ecg['ecg'][0][50:140]
    elif type_s is 'fast':
        x_1 = x['x'][0][:]
        ecg_1 = ecg['ecg'][0][:]
    elif type_s is 'slow':
        x_1 = x['x'][0][:]
        ecg_1 = ecg['ecg'][0][:]

    plt.figure(1)
    plt.title("EKG")
    plt.plot(x_1,ecg_1)
    #plt.show()
    
    return ecg_1,x_1 

def dft(sig,fs,bins=512):
    zero_pad = (10**math.ceil(math.log(sig.shape[0],2)) - sig.shape[0])
    sig = np.append(sig, np.zeros(zero_pad))
    out = np.fft.fft(sig, n=bins)
    freq_axis_scale = (fs)/bins
    #each bin represents 10 Hz
    out_abs = np.abs(out.real)
    plt.figure(6)
    plt.title("ECG Power Spectral Density")
    plt.plot(np.arange(freq_axis_scale,(fs/2)+freq_axis_scale,freq_axis_scale),out_abs[0:bins//2 ])
    #plt.xticks(np.arange(0,(fs)+1000,round(freq_axis_scale)))
    plt.xlabel('Hz')
    plt.ylabel("|H(e^jw)|")
    plt.show()

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

def notch_filt(w0,Q,fs):
    return sg.iirnotch(w0,Q,fs)


def bandpass(fs):
    r1 = 47e3
    r2 = 1e6
    c1 = .33e-6
    c2 = 22e-9
    num = [c1*r2, 0]
    den = [c1*c2*r1*r2, c1*r2+c2*r1-c1*r1, 1]
    h_s = sg.TransferFunction(num,den)
    w_s = np.linspace(.1*2*math.pi,300*2*math.pi,5000)
    w,mag,phase = sg.bode(h_s,w=w_s)

    plt.figure(2)
    plt.title("Magnitude")
    plt.semilogx(w_s/(2*math.pi),mag)
    plt.axvline(x=.5,color='orange', linestyle='--',label='fc_1(.5 Hz)')
    plt.axvline(x=153,color='orange', linestyle='--',label='fc_2(153 Hz)')
    plt.legend(loc='upper right')
    plt.xlabel("Freq. [Hz]")
    plt.ylabel("Mag. [dB]")
    plt.figure(3)
    plt.title("Phase")
    plt.semilogx(w_s/(2*math.pi),phase)
    plt.axvline(x=.5,color='orange', linestyle='--',label='fc_1(.5 Hz)')
    plt.axvline(x=153,color='orange', linestyle='--',label='fc_2(153 Hz)')
    plt.legend(loc='upper right')
    plt.xlabel("Freq. [Hz]")
    plt.ylabel("[deg]")
    
    #Discrete conversion
    w_d = np.linspace(.1*2*math.pi,300*2*math.pi,5000)
    num_d = [.3871, -.3871]
    den_d = [1, -1.629, .6298]
    h_s_d = sg.TransferFunction(num_d,den_d,dt=1/fs)
    w_d,mag_d,phase_d = sg.dbode(h_s_d,w=w_d/fs)
    #z,p,k = sg.tf2zpk(num,den)
    #w_d,mag_d,phase_d = sg.dbode((z,p,k,1/Fs),w=w_d/Fs)

    plt.figure(4)
    plt.title("Magnitude discrete")
    plt.semilogx(w_d/(2*math.pi),mag_d)
    plt.axvline(x=.5,color='orange', linestyle='--',label='fc_1(.5 Hz)')
    plt.axvline(x=153,color='orange', linestyle='--',label='fc_2(153 Hz)')
    plt.legend(loc='upper right')
    plt.xlabel("Freq. [Hz]")
    plt.ylabel("Mag. [dB]")
    plt.figure(5)
    plt.title("Phase discrete")
    plt.semilogx(w_d/(2*math.pi),phase_d)
    plt.axvline(x=.5,color='orange', linestyle='--',label='fc_1(.5 Hz)')
    plt.axvline(x=153,color='orange', linestyle='--',label='fc_2(153 Hz)')
    plt.legend(loc='upper right')
    plt.xlabel("Freq. [Hz]")
    plt.ylabel("[deg]")
    plt.show()

    return h_s_d, num_d, den_d

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    plt.figure(6)
    plt.title("Bandpass PZ plot")
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
    

    #return z, p, k



#For mac
#file_ecg = "/Users/pproctor/Documents/PSU/ECE522/project/default_ecg.mat"
#file_x = "/Users/pproctor/Documents/PSU/ECE522/project/default_ecg_x.mat"
#For Linux
file_ecg = "/u/pproctor/Documents/psu_courses/ECE522/ekg_project/default_ecg.mat"
file_x = "/u/pproctor/Documents/psu_courses/ECE522/ekg_project/default_ecg_x.mat"
Fs = 2000


data, t = load_mat(file_ecg,file_x)
data = data - np.mean(data)
plt.figure(1)
plt.title("EKG")
plt.plot(t,data)

dft(data,Fs)


H_d,b_d,a_d = bandpass(Fs)

#zplane(b_d,a_d)
plt.show()

sig_filt = sg.lfilter(b_d,a_d,data)
#sig_filt = sg.filtfilt(b_d,a_d,data)
#sig_filt = sig_filt - np.mean(sig_filt)
dft(sig_filt,Fs)

plt.figure(8)
plt.plot(t,sig_filt,c='orange')
plt.plot(t,data)
plt.show()

#sig_filt_con = sg.lfilter(b_d,a_d,sg.unit_impulse(90))
#plt.figure(9)
#plt.plot(t,sig_filt_con)
#plt.show()




