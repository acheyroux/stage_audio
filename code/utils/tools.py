import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, istft
from IPython.display import Audio, display
from metrics import SNR
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio

#Recherche GPU/CPU
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")




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
    
    '''
    maxi=50
    
    Sy=np.abs(y_Sgram)**2
    Sb=white_noise_DSP(sigma=sigma,signal_length=len(y),fs=fs)
    Sx=np.maximum(Sy-threshold*maxi*Sb,0)
    H=Sx/Sy
    #mask=np.abs(y_Sgram)>=threshold*sigma**2
    y_Sgram_mask= np.sqrt(H)*y_Sgram
    '''
    y_t,y_f,y_Sgram=spectrogram(y,fs)
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
from torch import nn
import deepinv as dinv

'''
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
'''

class NumpyDenoiserWrapper(nn.Module):
    def __init__(self, denoise_func, sigma, param, fs):
        super().__init__()
        self.denoise_func = denoise_func
        self.sigma = float(sigma)
        self.param = float(param)
        self.fs = int(fs)

    def forward(self, y, *args, **kwargs):
        input_device = y.device
        input_dtype = y.dtype

        y_cpu = y.detach().cpu().numpy()
        out = np.zeros_like(y_cpu)

        batch_size = y_cpu.shape[0]

        for i in range(batch_size):
            signal_1d = y_cpu[i, 0, 0, :]

            _, denoised_1d = self.denoise_func(
                signal_1d,
                self.sigma,
                self.param,
                self.fs
            )

            denoised_1d = np.asarray(denoised_1d, dtype=y_cpu.dtype)

            if len(denoised_1d) > len(signal_1d):
                denoised_1d = denoised_1d[:len(signal_1d)]
            elif len(denoised_1d) < len(signal_1d):
                denoised_1d = np.pad(
                    denoised_1d,
                    (0, len(signal_1d) - len(denoised_1d)),
                    mode="constant"
                )

            out[i, 0, 0, :] = denoised_1d

        return torch.from_numpy(out).to(device=input_device, dtype=input_dtype)
def deepinv_sure_param_search(
    y_np,
    param_values,
    sigma,
    fs,
    denoise_func,
    chunk_size=16384,
    device=None,
    tau=0.01,
    n_sure_repeats=1,
    param_name="param",
):
    """
    Finds the SURE-minimizing value of the third denoiser parameter.

    The denoise function must have the format:
        denoise_func(y, sigma, param, fs) -> (t_denoise, y_denoise)

    Examples
    --------
    Spectral subtraction:
        param = threshold

    Demucs:
        param = drywet

    SGMSE:
        param = N, if your SGMSE function interprets the third argument as N
    """

    if device is None:
        device = dinv.utils.get_device()

    y_np = np.asarray(y_np, dtype=np.float32)

    if y_np.ndim != 1:
        raise ValueError("y_np must be a mono 1D NumPy array.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    param_values = np.asarray(param_values, dtype=float)

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

    for param in param_values:

        model = NumpyDenoiserWrapper(
            denoise_func=denoise_func,
            sigma=sigma,
            param=param,
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
    best_param = float(param_values[best_idx])

    print("SURE best index:", best_idx, "/", len(param_values) - 1)
    print(f"SURE best {param_name}:", best_param)
    print("SURE first value:", sure_values[0])
    print("SURE best value:", sure_values[best_idx])
    print("SURE last value:", sure_values[-1])

    if best_idx == len(param_values) - 1:
        print(f"WARNING: SURE selected the maximum tested {param_name}.")

    if best_idx == 0:
        print(f"WARNING: SURE selected the minimum tested {param_name}.")

    best_model = NumpyDenoiserWrapper(
        denoise_func=denoise_func,
        sigma=sigma,
        param=best_param,
        fs=fs
    ).to(device)

    with torch.no_grad():
        x_best = best_model(y_tensor)

    denoised_chunks = x_best.detach().cpu().numpy()[:, 0, 0, :]
    best_denoised = denoised_chunks.reshape(-1)[:original_len]

    return best_denoised, best_param, sure_values



#Debruiteurs Deep
#DEMUCS
_DEVICE=device
_MODEL=pretrained.dns64()
_MODEL.to(_DEVICE)
_MODEL.eval()
_MODEL_SR=int(_MODEL.sample_rate)

def demucs_denoise(y,sigma,drywet,fs):
    _=sigma

    fs=int(fs)
    drywet=float(drywet)
    y=np.asarray(y,dtype=np.float32).squeeze()

    y_torch=torch.from_numpy(y).float()[None,:]
    y_model_sr=convert_audio(y_torch,fs,_MODEL_SR,1)

    y_model_sr=y_model_sr.to(_DEVICE)

    with torch.no_grad():
        y_hat=_MODEL(y_model_sr[None])

    y_demucs=y_hat[0,0].detach().cpu().numpy()
    y_noisy=y_model_sr[0].detach().cpu().numpy()

    min_len=min(len(y_demucs),len(y_noisy))
    y_demucs=y_demucs[:min_len]
    y_noisy=y_noisy[:min_len]

    y_denoise=drywet*y_demucs+(1.0-drywet)*y_noisy
    t_denoise=np.arange(len(y_denoise))/_MODEL_SR

    return t_denoise,y_denoise


'''
#DEMUCS
from denoiser import pretrained #DEMUCS
from denoiser.dsp import convert_audio


def denoise_demucs(y,sigma,threshold,fs)
    #Detection de GPU/CPU
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    #Import du fichier sonore et modele DEMUCS
    sound,samplerate=y,fs
    
    model = getattr(pretrained, params['demucs_pretrained_model'])()
    model = model.to(device)
    model.eval()
    model_sample_rate=model.sample_rate

    sound_t=torch.from_numpy(sound).float()[None, :]
    sound=convert_audio(sound_tensor,samplerate,model_sample_rate,1)[0].detach().cpu().numpy()

    
    sound_t=convert_audio(torch.from_numpy(sound).float()[None, :],samplerate,model_sample_rate,1)[0].detach().cpu().numpy()

    
    with torch.no_grad():



        #Debruitage DEMUCS
        son_bt = torch.from_numpy(son_b).float()[None, :].to(device)
        son_dt=model(son_bt[None])
        y_denoise=son_dt[0, 0].detach().cpu().numpy()

    return t_denoise,y_denoise
    '''