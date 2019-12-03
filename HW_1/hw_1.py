import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Todo:
Plot values of ewma_signal
Fix the initial edge before window is fulfilled
Calculate the average annual return
Plot everything
"""

def return_eqn(y_f, y_i, n_f):
    return ((y_f / y_i)) ** (1/n_f) - 1

def slide_win(source_d,close_d,return_d, num_years):
    idx = 0
    for idx in range(source_d.shape[0]):
        if idx == 4959:
            break
        for idx_in in range(255 * num_years):
            if source_d[idx][5:] == source_d[(idx_in+idx)+1][5:]:
                break
            elif source_d[idx+1][5:] == source_d[(idx_in+idx)+2][5:]:
                break
            return_d[idx][idx_in] = close_d[idx_in+idx]

    return return_d

def ann_return(return_data, source_d):
    #Calc. annualized return
    for jj in range(4959):
        y_i = source_d[jj][0]
        if source_d[jj][246] == 0:
            y_f = source_d[jj][245]    
        elif source_d[jj][247] == 0:
            y_f = source_d[jj][246]
        elif source_d[jj][248] == 0:
            y_f = source_d[jj][247]
        elif source_d[jj][249] == 0:
            y_f = source_d[jj][248]    
        elif source_d[jj][250] == 0:
            y_f = source_d[jj][249]
        elif source_d[jj][251] == 0:
            y_f = source_d[jj][250]
        elif source_d[jj][252] == 0:
            y_f = source_d[jj][251]    
        elif source_d[jj][253] == 0:
            y_f = source_d[jj][252]
        elif source_d[jj][254] == 0:
            y_f = source_d[jj][253]
        else:
            y_f = source_d[jj][254]
    
        return_data[jj] = return_eqn(y_f,y_i,1)
    return return_data


def hist_percent(input_arr):
    #plt.figure(2)
    perc_5 = np.percentile(input_arr,5)
    plt.figure(2)
    plt.hist(input_arr, bins='auto')
    return perc_5

def ewma_filt(signal,window):
    ewma_signal = np.zeros(signal.shape[0])
    alpha = .95#2 /(window + 1.0)
    alpha_rev = 1-alpha

    pows = alpha_rev**(np.arange(window+1))

    offset = signal[0]*pows[1:]
    ewma_signal[0:window] = offset
    ewma_signal = np.convolve(signal,pows, mode='same')
    #ewma_signal[1:] = np.cumsum()
    pw0 = alpha_rev**(window-1)
    """
    y[0] = offset
    y[n] = a*x[n] + (1-a)^n*y[n-1]
    """
    """
    for jj in range(signal.shape[0]):
        if jj < window:
            ewma_signal[0:window] = offset
        elif jj >= window:
            ewma_signal[jj] = np.cumsum(alpha*signal[jj] + alpha_rev * pows[1:] * ewma_signal[(jj-window):jj])[-1:][0]
    """
    #mult = signal*pw0*scale_arr
    #out = offset + np.cumsum*scale_arr[::-1]
    return ewma_signal

#file_p = '/u/pproctor/Documents/psu_courses/EE522/HW_1/NVDA.csv'
file_p = os.path.abspath('/Users/pproctor/Documents/PSU/ECE522/HW_1/NVDA.csv')
fin_data = pd.read_csv(file_p)

close_d = fin_data['Close'].to_numpy()
open_d = fin_data['Open'].to_numpy()
date_d = fin_data['Date'].to_numpy()



"""
Find date that matches a year later, calculate annualized return
Slide one step, find date that matches a year later, calculate annualized return
"""

num_years = 1
return_arr = np.zeros(4959)
close_d_ch = np.ndarray(shape=(4959,255), dtype=float)
close_d_ch.fill(0)
close_d_ch = slide_win(date_d,close_d,close_d_ch, num_years)

return_arr = ann_return(return_arr,close_d_ch)
percentile_5 = hist_percent(return_arr)
#freq_hist = freq(return_arr)

#Filtering
window = 50
close_filt_d = np.convolve(close_d, np.ones((window,))/window, mode='same')

window_2 = 5
ewma_filt_d = ewma_filt(close_d,window_2)

plt.figure(1)
plt.plot(range(date_d.shape[0]), close_d, c='black', label='Raw')
#plt.plot(range(date_d.shape[0]), close_filt_d, c='red', label='Filtered')
plt.plot(range(date_d.shape[0]),ewma_filt_d)
plt.grid()
plt.show()

print("None")
