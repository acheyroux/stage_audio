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