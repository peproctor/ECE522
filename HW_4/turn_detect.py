import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig_p
import os

def plots(labels,data,data_f,idx,fs,s1_idx):
    
    plt.figure(1)
    plt.subplot(141)
    plt.plot(s1_idx,data[labels[0]][:idx])
    plt.plot(s1_idx,data_f[labels[0]][:idx], c='orange')
    plt.title(labels[0])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.subplot(142)
    plt.plot(s1_idx,data[labels[1]][:idx])
    plt.title(labels[1])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.subplot(143)
    plt.plot(s1_idx,data[labels[2]][:idx])
    plt.title(labels[2])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    
    plt.subplot(144)
    plt.plot(s1_idx,data[labels[3]][:idx])
    plt.title(labels[3])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.tight_layout()

    plt.figure(2)
    plt.subplot(131)
    plt.plot(s1_idx,data[labels[4]][:idx])
    plt.title(labels[4])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.subplot(132)
    plt.plot(s1_idx,data[labels[5]][:idx])
    plt.title(labels[5])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.subplot(133)
    plt.plot(s1_idx,data[labels[6]][:idx])
    plt.title(labels[6])
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [rad/sec]')

    plt.tight_layout()

def plot_solo(labels,data,data_f,idx,fs,s1_idx):
        plt.figure(1)
        plt.plot(s1_idx,data[dict_labels[0]][:idx])
        plt.plot(s1_idx,data_f[dict_labels[0]][:idx], c='orange')
        plt.title(dict_labels[0])
        plt.xlabel('Time [s]')
        plt.ylabel('Vel [rad/sec]')

        plt.figure(2)
        plt.plot(s1_idx,data[dict_labels[1]][:idx])
        plt.plot(s1_idx,data_f[dict_labels[1]][:idx], c='orange')
        plt.title(dict_labels[1])
        plt.xlabel('Time [s]')
        plt.ylabel('Vel [rad/sec]')

        plt.figure(3)
        plt.plot(s1_idx,data[dict_labels[2]][:idx])
        plt.plot(s1_idx,data_f[dict_labels[2]][:idx], c='orange')
        plt.title(dict_labels[2])
        plt.xlabel('Time [s]')
        plt.ylabel('Vel [rad/sec]')

        plt.figure(4)
        plt.plot(s1_idx,data[dict_labels[3]][:idx])
        plt.plot(s1_idx,data_f[dict_labels[3]][:idx], c='orange')
        plt.title(dict_labels[3])
        plt.xlabel('Time [s]')
        plt.ylabel('Vel [rad/sec]')

def SMA_filt(signal,size,fs, coeff=True):
    # SMA coefficients
    b = np.ones(size)
    a = np.array([size] + [0]*(size-1))
    #bode(b,a,fs)
    if coeff is True:
        return sig_p.filtfilt(b,a,signal)
    else:
            return b,a


def bode(num,den,fs):
    w,mag,phase = sig_p.dbode((num,den,1/fs), n=1000)
    plt.figure(5)
    plt.subplot(121)
    plt.title('Mag')
    plt.plot((w/(2*np.pi)),mag)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude [dB]')

    plt.subplot(122)
    plt.title('Phase')
    plt.plot((w/(2*np.pi)),phase)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Phase [deg]')
 
def trap_int(x,y,a,b):
        h = x[a+1] - x[a]
        intgr = .5*h*(y[b]+y[a])
        for i in range(a,b):
                intgr = intgr + h*y[i]

        return intgr

def bound_extract(data,x,fs,N):
        #np.random.seed(1)
        #rand_idx = np.random.randint(0,data.shape[0],size=int((N*.5)))
        plt.figure(9)
        plt.hist(data,bins='auto')
        

        thresh = np.mean(np.abs(data))
        peaks,_ = sig_p.find_peaks(data, height=(thresh,None))
        plt.figure(10)
        plt.plot(x_vals*(1/fs),data)
        plt.scatter(peaks*(1/fs),data[peaks],c='red')
        plt.show()
        b = 0
        a = 0
        return b,a, peaks

fs = 128 #Hz
T = 1/fs 
wind = 50

data_dict = {}
filt_dict = {}
idx_ls = 16128
x_vals = np.arange(idx_ls) 

"""
To do:
Determine thresholding method for peak finding
Determine method to find integration bounds
Integrate bounds
"""

file_p = '/Users/pproctor/Documents/PSU/ECE522/HW_4'
file_ls = os.listdir(file_p)
zz = 0
for da_file in file_ls:
    if (da_file[-4:] == '.txt'):
        data_dict[da_file[:-4]] = np.loadtxt(os.path.join(file_p,da_file))
        filt_dict[da_file[:-4]] = SMA_filt(data_dict[da_file[:-4]],wind,fs)
        zz+=1

dict_labels = list(data_dict.keys())
dict_labels.sort()


plot_solo(dict_labels,data_dict,filt_dict,idx_ls,fs,x_vals* (1/fs))
#plt.show()

b,a,peaks = bound_extract(filt_dict[dict_labels[0]],x_vals,fs,idx_ls)

#b = 2700
#a = 2204
integral = trap_int(x_vals,data_dict[dict_labels[0]],a,b)
int_np = np.trapz(data_dict[dict_labels[0]][a:b],x_vals[a:b])

#plots(dict_labels,data_dict,filt_dict,idx_ls,fs, x_vals)

num,den = SMA_filt(np.ones(10),wind,fs,False)
bode(num,den,fs)
plt.show()

print("Hello")
