import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, istft
from IPython.display import Audio, display

def signal(x,fs,show=False):
    '''
    Retourne le signal avec son axe temporelle
    --- In ---
    x : signal (ndarray)
    fs : frequence d'echantillonage (float)
    show : trace le signal temporel (bool)
    --- Out ---
    t : axe temporel (ndarray)
    x : signal (ndarray)
    '''
    t=np.arange(len(x))/fs
    if show:
        plt.plot(t,x)
        plt.title('Signal')
        plt.ylabel('Amplitude')
        plt.xlabel('Time [sec]')
        plt.show()
    return t,x

def periodogram(x, fs: float,show=False):
    '''
    Periodogramme d'un signal entree
    --- In ---
    x : signal (ndarray)
    fs : frequence d'echantillonage (float)
    show : trace le periodogramme (bool)
    --- Out ---
    f : axe frequentiel (ndarray)
    Pgram : Densite spectrale DSP (ndarray)
    '''
    x_hat=np.fft.fft(x)
    f=np.fft.fftfreq(len(x),1/fs)
    Pgram=(np.abs(x_hat)**2)/(len(x)*fs)
    if show:
        plt.plot(f,Pgram,label='Spectral Density')
        plt.title('Periodogram')
        plt.ylabel('Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.show()
    return f, Pgram

def spectrogram(x, fs: float, show=False):
    '''
    Trace le spectrogramme du signal d'entree
    --- In ---
    x : signal (ndarray)
    fs : frequence d'echantillonage (float)
    show : trace le periodogramme (bool)
    --- Out ---
    t : axe temporel
    f : axe frequentiel
    Sgram : de
    '''
    f,t,Sgram=stft(x,fs,window='hann',nperseg=1024,noverlap=256,boundary="zeros",padded=True,)
    if show:
        plt.pcolormesh(t, f, np.abs(Sgram))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time Index')
        plt.colorbar()
        plt.show()
    return t,f,Sgram

def denoise_spectral_sub(y, sigma, threshold, fs):
    '''
    Retourne le signal apres soustraction spectrale
    --- In ---
    y : signal (ndarray)
    sigma : ecart type du bruit (float)
    threshold : seuil de coupure (float)
    fs : frequence d'echantillonage (float)
    show : trace le signal temporel (bool)
    --- Out ---
    x : signal apres soustraction spectrale (ndarray)
    '''
    y_t,y_f,y_Sgram=spectrogram(y,fs)
    mask=np.abs(y_Sgram)>=threshold*sigma**2
    y_Sgram_mask= y_Sgram*mask
    t_denoise,y_denoise=istft(y_Sgram_mask,fs,window='hann',nperseg=1024,noverlap=256,boundary=True)
    return t_denoise,y_denoise

def listen(audio,samplerate):
    '''
    Affiche le lecteur sonore pour le signal d'entree
    --- In ---
    audio : signal (ndarray)
    samplerate : frequence d'echantillonage (float)
    --- Out ---
    None
    '''
    display(Audio(audio, rate=samplerate))
    return

def find_oracle_threshold(x,xb,sigma,fs,denoise_func):
    '''
    Retourne le seuil oracle 
    --- In ---
    x : signal pur (ndarray)
    xb : signal bruite (ndarray)
    sigma : ecart type du bruit gaussien
    fs : frequence d'echantillonage (float)
    denoise_func : fonction de debruitage sous forme (y, sigma, threshold, fs) --> (t_denoise, y_denoise)
    --- Out ---
    oracle_threshold : seuil oracle du signal (float)
    '''
    thresholds=np.linspace(0,3,100)
    oSNR_max=-np.inf
    oracle_threshold=0
    for thresh in thresholds:
        _,x_d=denoise_func(xb,sigma,thresh,fs)
        max_len=min(len(x),len(x_d))
        oSNR=SNR(x[:max_len],(x_d[:max_len]-x[:max_len]))
        if oSNR>=oSNR_max:
            oSNR_max=oSNR
            oracle_threshold=thresh
    return oracle_threshold