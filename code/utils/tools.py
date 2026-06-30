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
    maxi=50
    y_t,y_f,y_Sgram=spectrogram(y,fs)
    Sy=np.abs(y_Sgram)**2
    Sb=white_noise_DSP(sigma=sigma,signal_length=len(y),fs=fs)
    Sx=np.maximum(Sy-threshold*maxi*Sb,0)
    H=Sx/Sy
    #mask=np.abs(y_Sgram)>=threshold*sigma**2
    y_Sgram_mask= np.sqrt(H)*y_Sgram
    
    lamb=threshold*np.max(y_Sgram)
    y_Sgram_mask=y_Sgram*np.maximum(1-(lamb/np.abs(y_Sgram))**2,np.ones_like(y_Sgram)*0)
    
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
    thresholds=np.linspace(0,40,100)
    alphas=np.linspace(0, 1, 100)
    #thresholds = alphas * np.linalg.norm(xb, ord=np.inf)
    thresholds=np.logspace(-3,0,100)
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
        return oracle_threshold,fig,thresholds,oSNR_list
    return oracle_threshold

#SURE (cf ChatGPT)
import numpy as np
import torch
from torch import nn
import deepinv as dinv


class SpectralSubNumpyWrapper(nn.Module):
    """
    Wraps your actual NumPy spectral subtraction function so DeepInv can call it.

    Your denoise function must look like:
        denoise_spectral_sub(y, sigma, threshold, fs) -> (t_denoise, y_denoise)

    Input tensor shape:
        [B, 1, 1, T]

    Output tensor shape:
        [B, 1, 1, T]
    """

    def __init__(self, denoise_func, sigma, threshold, fs):
        super().__init__()
        self.denoise_func = denoise_func
        self.sigma = float(sigma)
        self.threshold = float(threshold)
        self.fs = float(fs)

    def forward(self, y, *args, **kwargs):
        device = y.device
        dtype = y.dtype

        y_cpu = y.detach().cpu().numpy()
        out = np.zeros_like(y_cpu)

        batch_size = y_cpu.shape[0]

        for i in range(batch_size):
            signal_1d = y_cpu[i, 0, 0, :]

            _, denoised_1d = self.denoise_func(
                signal_1d,
                self.sigma,
                self.threshold,
                self.fs
            )

            denoised_1d = np.asarray(denoised_1d, dtype=y_cpu.dtype)

            # Force same length as input
            if len(denoised_1d) > len(signal_1d):
                denoised_1d = denoised_1d[:len(signal_1d)]
            elif len(denoised_1d) < len(signal_1d):
                denoised_1d = np.pad(
                    denoised_1d,
                    (0, len(signal_1d) - len(denoised_1d)),
                    mode="constant"
                )

            out[i, 0, 0, :] = denoised_1d

        return torch.from_numpy(out).to(device=device, dtype=dtype)


def deepinv_sure_spectral_sub_threshold_search(
    y_np,
    thresholds,
    sigma,
    fs,
    denoise_func,
    chunk_size=16384,
    device=None,
    tau=0.01,
    n_sure_repeats=1,
):
    """
    Finds the SURE-minimizing threshold using your actual spectral subtraction
    function denoise_spectral_sub.
    """

    if device is None:
        device = dinv.utils.get_device()

    y_np = np.asarray(y_np, dtype=np.float32)

    if y_np.ndim != 1:
        raise ValueError("y_np must be a mono 1D NumPy array.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    thresholds = np.asarray(thresholds, dtype=float)

    original_len = len(y_np)

    pad_len = (-original_len) % chunk_size

    if pad_len > 0:
        y_padded = np.pad(y_np, (0, pad_len), mode="reflect")
    else:
        y_padded = y_np

    chunks = y_padded.reshape(-1, chunk_size)

    # DeepInv expects image-like tensors: [B, C, H, W]
    # For 1D audio, we use [chunks, 1, 1, chunk_size]
    y_tensor = torch.from_numpy(chunks[:, None, None, :]).to(device)

    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(sigma=float(sigma))
    )

    sure_loss = dinv.loss.SureGaussianLoss(
        sigma=float(sigma),
        tau=float(tau)
    )

    sure_values = []

    for threshold in thresholds:

        model = SpectralSubNumpyWrapper(
            denoise_func=denoise_func,
            sigma=sigma,
            threshold=threshold,
            fs=fs
        ).to(device)

        x_net = model(y_tensor)

        repeated_losses = []

        for _ in range(n_sure_repeats):
            loss = sure_loss(
                x_net=x_net,
                y=y_tensor,
                physics=physics,
                model=model,
            )

            repeated_losses.append(loss.detach().mean().cpu().item())

        sure_scalar = float(np.mean(repeated_losses))
        sure_values.append(sure_scalar)

    sure_values = np.asarray(sure_values)

    best_idx = int(np.argmin(sure_values))
    best_threshold = float(thresholds[best_idx])

    print("SURE best index:", best_idx, "/", len(thresholds) - 1)
    print("SURE best threshold:", best_threshold)
    print("SURE first value:", sure_values[0])
    print("SURE best value:", sure_values[best_idx])
    print("SURE last value:", sure_values[-1])

    if best_idx == len(thresholds) - 1:
        print("WARNING: SURE selected the maximum tested threshold.")

    best_model = SpectralSubNumpyWrapper(
        denoise_func=denoise_func,
        sigma=sigma,
        threshold=best_threshold,
        fs=fs
    ).to(device)

    with torch.no_grad():
        x_best = best_model(y_tensor)

    denoised_chunks = x_best.detach().cpu().numpy()[:, 0, 0, :]
    best_denoised = denoised_chunks.reshape(-1)[:original_len]

    return best_denoised, best_threshold, sure_values