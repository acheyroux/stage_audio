import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, istft
from IPython.display import Audio, display
from metrics import SNR

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
    f,t,Sgram=stft(x,fs,window='hann',nperseg=1024,noverlap=768,boundary="zeros",padded=True,)
    if show:
        plt.pcolormesh(t, f, np.abs(Sgram))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time Index')
        plt.colorbar()
        plt.show()
    return t,f,Sgram

def white_noise_DSP(sigma, signal_length, fs):
    """
    Estime la DSP du bruit blanc d'ecart sigma
    --- In ---
    sigma : ecart du bruit blanc (float)
    signal_length : longueur du signal (int)
    fs : frequence d'echantillonage (float)
    --- Out ---
    Sb : DSP estimee du bruit (ndarray)
    """
    np.random.seed(0)
    b=sigma*np.random.standard_normal(signal_length)
    _,_,B=stft(b,fs=fs,window="hann",nperseg=1024,noverlap=768,boundary="zeros",padded=True)
    Sb=np.mean(np.abs(B)**2,axis=1,keepdims=True)
    return Sb


def denoise_spectral_sub(y,sigma,threshold,fs):
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
    
    Sy=np.abs(y_Sgram)**2
    Sb=white_noise_DSP(sigma=sigma,signal_length=len(y),fs=fs)
    Sx=np.maximum(Sy-threshold*Sb,0)
    H=Sx/Sy
    #mask=np.abs(y_Sgram)>=threshold*sigma**2
    y_Sgram_mask= H*y_Sgram
    t_denoise,y_denoise=istft(y_Sgram_mask,fs,window='hann',nperseg=1024,noverlap=768,boundary=True)
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

def find_oracle_threshold(x,xb,sigma,fs,denoise_func,graph=False):
    '''
    Retourne le seuil oracle 
    --- In ---
    x : signal pur (ndarray)
    xb : signal bruite (ndarray)
    sigma : ecart type du bruit gaussien
    fs : frequence d'echantillonage (float)
    denoise_func : fonction de debruitage sous forme (y, sigma, threshold, fs) --> (t_denoise, y_denoise)
    graph : export ou non du graphique (bool)
    --- Out ---
    oracle_threshold : seuil oracle du signal (float)
    if graph --> fig : graph du seuil (pyplot)
    '''
    thresholds=np.linspace(0,10,100)
    oSNR_max=-np.inf
    oSNR_list=[]
    oracle_threshold=0
    for thresh in thresholds:
        _,x_d=denoise_func(xb,sigma,thresh,fs)
        #max_len=min(len(x),len(x_d))
        oSNR=SNR(x,x_d)
        oSNR_list.append(oSNR)
        if oSNR>=oSNR_max:
            oSNR_max=oSNR
            oracle_threshold=thresh
    if graph:
        fig, ax = plt.subplots()
        ax.plot(thresholds, oSNR_list,label=r'$\sigma = $'+str(sigma))
        ax.set_xlabel("Threshold")
        ax.set_ylabel("oSNR")
        ax.set_title("Seuil Oracle")
        return oracle_threshold,fig
    return oracle_threshold