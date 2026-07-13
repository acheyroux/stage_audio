import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, istft
from IPython.display import Audio, display
from metrics import SNR
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os
import gc

import sgmse
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec
from sgmse.sampling import PredictorRegistry, CorrectorRegistry

#PyTorch
from torch import nn
import deepinv as dinv



#Recherche GPU/CPU
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")

print("Chosen device"+str(device))

#Fonction pour lecture de parametres.txt
def read_params(parameters_txt):
    params = {}
    for full_line in parameters_txt:
        line = full_line.strip()
        if not line[0] == '#':
            key, value = line.split('=', 1)
            val = value.strip().split(',')
            if len(val) != 1:
                for i in range(len(val)):
                    val[i] = float(val[i].strip())
            else:
                val = val[0].strip()
                if val.isdigit():
                    val = int(val)
                elif val == 'True':
                    val = True
                elif val == 'False':
                    val = False
                else:
                    try:
                        val = float(val)
                    except:
                        val = val
            params[key.strip()] = val
    
    if type(params['isnr']) != list:
        params['isnr'] = [params['isnr']]
    return params



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

    '''

def _demucs_forward_tensor_chunks(y_tensor, fs):
    """
    Runs DEMUCS once on a batch of chunks.

    Input:
        y_tensor: torch tensor with shape [B, 1, 1, T]
    Output:
        y_demucs_tensor: torch tensor with shape [B, 1, 1, T]
    """
    fs = int(fs)

    if y_tensor.ndim != 4:
        raise ValueError("y_tensor must have shape [B, 1, 1, T].")

    if y_tensor.shape[1] != 1 or y_tensor.shape[2] != 1:
        raise ValueError("Expected y_tensor shape [B, 1, 1, T].")

    input_device = y_tensor.device
    input_dtype = y_tensor.dtype

    # DEMUCS expects [B, C, T]
    y_demucs_in = y_tensor[:, 0, :, :]  # [B, 1, T]

    # Your current experiment already converts audio to 16 kHz before DEMUCS.
    # This fast path avoids slow per-chunk resampling.
    if fs != _MODEL_SR:
        raise ValueError(
            f"This cached DEMUCS SURE function expects fs == _MODEL_SR. "
            f"Got fs={fs}, but DEMUCS model sample rate is {_MODEL_SR}. "
            f"Resample your audio to {_MODEL_SR} before calling this function."
        )

    y_demucs_in = y_demucs_in.to(_DEVICE)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(_DEVICE.type == "cuda")):
            y_hat = _MODEL(y_demucs_in)

    # y_hat shape should be [B, 1, T_out]
    y_hat = y_hat.to(device=input_device, dtype=input_dtype)

    target_len = y_tensor.shape[-1]

    if y_hat.shape[-1] > target_len:
        y_hat = y_hat[..., :target_len]
    elif y_hat.shape[-1] < target_len:
        pad_len = target_len - y_hat.shape[-1]
        y_hat = torch.nn.functional.pad(y_hat, (0, pad_len))

    return y_hat[:, :, None, :]  # [B, 1, 1, T]



#Optimisation GPU Torch pour recherche SURE
def deepinv_sure_spectral_sub_threshold_search_torch(
    y_np,
    thresholds,
    sigma,
    fs=None,
    chunk_size=65536,
    device=None,
    tau=0.01,
    n_sure_repeats=1,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
):
    """
    GPU-optimized SURE threshold search for spectral subtraction.

    This version:
        - avoids SciPy
        - avoids NumPy inside the threshold loop
        - evaluates all thresholds in one batched torch.stft / torch.istft pass
        - keeps the whole SURE search on CUDA

    Parameters
    ----------
    y_np:
        1D mono noisy signal as NumPy array.

    thresholds:
        Threshold values to test.

    sigma:
        Assumed Gaussian noise std for SURE.

    fs:
        Unused, included only for API compatibility.

    Returns
    -------
    best_denoised:
        NumPy array, best denoised signal.

    best_threshold:
        Float, SURE-selected threshold.

    sure_values:
        NumPy array of SURE values for all thresholds.
    """

    _ = fs

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    y_np = np.asarray(y_np, dtype=np.float32)

    if y_np.ndim != 1:
        raise ValueError("y_np must be a mono 1D NumPy array.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    thresholds = np.asarray(thresholds, dtype=np.float32)

    original_len = len(y_np)

    pad_len = (-original_len) % chunk_size

    if pad_len > 0:
        y_padded = np.pad(y_np, (0, pad_len), mode="reflect")
    else:
        y_padded = y_np

    chunks = y_padded.reshape(-1, chunk_size)

    y_tensor = torch.from_numpy(chunks[:, None, None, :]).to(device)

    thresholds_t = torch.as_tensor(
        thresholds,
        device=device,
        dtype=y_tensor.dtype,
    )

    P = len(thresholds)
    n_per_chunk = y_tensor[0].numel()

    sure_accum = torch.zeros(
        P,
        device=device,
        dtype=y_tensor.dtype,
    )

    # A(y) for all thresholds.
    with torch.no_grad():
        x_all = spectral_sub_torch_thresholds(
            y_tensor,
            thresholds_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )  # [P, B, 1, 1, T]

    for _ in range(n_sure_repeats):
        b = torch.randn_like(y_tensor)
        y_perturbed = y_tensor + tau * b

        with torch.no_grad():
            x_perturbed_all = spectral_sub_torch_thresholds(
                y_perturbed,
                thresholds_t,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )  # [P, B, 1, 1, T]

        residual_term = (x_all - y_tensor[None]).pow(2)
        residual_term = residual_term.flatten(2).sum(dim=2)
        residual_term = residual_term / n_per_chunk  # [P, B]

        divergence_term = b[None] * (x_perturbed_all - x_all) / tau
        divergence_term = divergence_term.flatten(2).sum(dim=2)
        divergence_term = divergence_term / n_per_chunk  # [P, B]

        sure_per_chunk = (
            residual_term
            - float(sigma) ** 2
            + 2.0 * float(sigma) ** 2 * divergence_term
        )  # [P, B]

        sure_accum += sure_per_chunk.mean(dim=1)

    sure_values = (sure_accum / n_sure_repeats).detach().cpu().numpy()

    best_idx = int(np.argmin(sure_values))
    best_threshold = float(thresholds[best_idx])

    print("SURE best index:", best_idx, "/", len(thresholds) - 1)
    print("SURE best threshold:", best_threshold)
    print("SURE first value:", sure_values[0])
    print("SURE best value:", sure_values[best_idx])
    print("SURE last value:", sure_values[-1])

    if best_idx == len(thresholds) - 1:
        print("WARNING: SURE selected the maximum tested threshold.")

    if best_idx == 0:
        print("WARNING: SURE selected the minimum tested threshold.")

    x_best = x_all[best_idx]  # [B, 1, 1, T]

    denoised_chunks = x_best.detach().cpu().numpy()[:, 0, 0, :]
    best_denoised = denoised_chunks.reshape(-1)[:original_len]

    return best_denoised, best_threshold, sure_values

def spectral_sub_torch_thresholds(
    y_tensor,
    thresholds,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    eps=1e-12,
):
    """
    Batched GPU spectral subtraction for many thresholds at once.

    Parameters
    ----------
    y_tensor:
        Torch tensor with shape [B, 1, 1, T]
    thresholds:
        1D array-like or torch tensor with P threshold values

    Returns
    -------
    x_all:
        Torch tensor with shape [P, B, 1, 1, T]
        where P = number of thresholds.
    """

    if y_tensor.ndim != 4:
        raise ValueError("y_tensor must have shape [B, 1, 1, T].")

    if y_tensor.shape[1] != 1 or y_tensor.shape[2] != 1:
        raise ValueError("Expected y_tensor shape [B, 1, 1, T].")

    device = y_tensor.device
    dtype = y_tensor.dtype

    thresholds = torch.as_tensor(
        thresholds,
        device=device,
        dtype=dtype,
    ).view(-1)

    B = y_tensor.shape[0]
    T = y_tensor.shape[-1]
    P = thresholds.numel()

    y_flat = y_tensor[:, 0, 0, :]  # [B, T]

    window = torch.hann_window(
        win_length,
        device=device,
        dtype=dtype,
    )

    Y = torch.stft(
        y_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )  # [B, F, N]

    mag = torch.abs(Y)  # [B, F, N]

    # Match your intended logic:
    # lamb = threshold * max(abs(STFT))
    max_mag = mag.amax(dim=(-2, -1), keepdim=True)  # [B, 1, 1]

    lamb = thresholds[:, None, None, None] * max_mag[None, :, :, :]  # [P, B, 1, 1]

    mask = 1.0 - (lamb / (mag[None, :, :, :] + eps)).pow(2)
    mask = torch.clamp(mask, min=0.0)  # [P, B, F, N]

    Y_masked = Y[None, :, :, :] * mask  # [P, B, F, N]

    Y_masked_flat = Y_masked.reshape(P * B, Y.shape[-2], Y.shape[-1])

    x_flat = torch.istft(
        Y_masked_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=T,
    )  # [P * B, T]

    x_all = x_flat.reshape(P, B, T)
    x_all = x_all[:, :, None, None, :]  # [P, B, 1, 1, T]

    return x_all




#New DEMUCS
def np_audio_to_tensor(audio_np: np.ndarray) -> torch.Tensor:
    """
    Converts numpy audio to torch tensor shaped [1, channels, samples].
    Accepts:
      [samples]
      [channels, samples]
      [samples, channels]
    """
    audio_np = np.asarray(audio_np, dtype=np.float32)

    if audio_np.ndim == 1:
        audio_np = audio_np[None, :] # [1, samples]

    elif audio_np.ndim == 2:
        # If shape is [samples, channels], transpose to [channels, samples]
        if audio_np.shape[0] > audio_np.shape[1]:
            audio_np = audio_np.T

    else:
        raise ValueError("audio_np must be 1D or 2D")

    audio = torch.from_numpy(audio_np).unsqueeze(0) # [1, C, T]
    return audio.to(device)


def tensor_audio_to_np(audio_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts [1, channels, samples] tensor back to numpy.
    """
    audio = audio_tensor.detach().cpu().squeeze(0).numpy()

    if audio.shape[0] == 1:
        return audio[0] # mono [samples]

    return audio # stereo/multichannel [channels, samples]


def sure_loss(model, noisy_audio, sigma, eps=1e-3):
    """
    SURE loss for Gaussian additive noise.

    noisy_audio: torch tensor [B, C, T]
    sigma: noise standard deviation, not variance
    """

    clean_estimate = model(noisy_audio)

    # Monte Carlo divergence approximation
    b = torch.randn_like(noisy_audio)

    clean_estimate_eps = model(noisy_audio + eps * b)

    divergence = torch.sum(
        b * (clean_estimate_eps - clean_estimate)
    ) / eps

    residual = torch.sum((clean_estimate - noisy_audio) ** 2)

    n = noisy_audio.numel()
    sigma2 = sigma ** 2

    loss = residual + 2.0 * sigma2 * divergence - n * sigma2

    return loss / n

class DemucsForDeepInvSURE(nn.Module):
    def __init__(self, demucs_model):
        super().__init__()
        self.demucs_model = demucs_model

    def forward(self, y, physics=None, *args, **kwargs):
        return self.demucs_model(y)

def demucs_sure_denoise(
    audio_np: np.ndarray,
    sigma: float,
    steps: int = 100,
    lr: float = 1e-6,
    eps: float = 1e-3,
    tau=1e-3,
    mc_batch_size: int = 4,
    debug=False,
):
    """
    Fine-tunes Demucs on one noisy audio sample using SURE.

    Each call:
        - starts from a fresh pretrained dns64 checkpoint
        - uses the full audio, not crops
        - batches multiple Monte Carlo SURE perturbations at once
    """

    local_demucs = pretrained.dns64()
    local_demucs.to(device)

    demucs_model = DemucsForDeepInvSURE(local_demucs).to(device)
    demucs_model.train()

    for param in demucs_model.parameters():
        param.requires_grad = True

    trainable_params = [p for p in demucs_model.parameters() if p.requires_grad]

    if debug:
        n_trainable = sum(p.numel() for p in trainable_params)
        print("Trainable Demucs parameters:", n_trainable)

    noisy_audio = np_audio_to_tensor(audio_np)  # [1, C, T]

    # Full-audio batch: same full signal repeated several times.
    # No cropping.
    noisy_batch = noisy_audio.repeat(mc_batch_size, 1, 1)  # [B, C, T]

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(sigma=float(sigma))
    )

    sure_loss_fn = dinv.loss.SureGaussianLoss(
        sigma=float(sigma),
        tau=float(tau)
    )

    if debug:
        with torch.no_grad():
            output_before = demucs_model(noisy_audio).detach().clone()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
    
        x_net = demucs_model(noisy_batch)
    
        loss_per_item = sure_loss_fn(
            x_net=x_net,
            y=noisy_batch,
            physics=physics,
            model=demucs_model,
        )
    
        loss = loss_per_item.mean()
    
        loss.backward()
        optimizer.step()
    
        print(
            f"Demucs SURE step {step + 1}/{steps} | "
            f"mean={loss.item():.6e} | "
            f"min={loss_per_item.min().item():.6e} | "
            f"max={loss_per_item.max().item():.6e}"
        )

        if debug:
            print(f"step {step:03d} | SURE={loss.item():.6e}")

    demucs_model.eval()

    with torch.no_grad():
        output_after = demucs_model(noisy_audio)

        if debug:
            output_change = torch.mean((output_after - output_before) ** 2).item()
            input_distance_before = torch.mean((output_before - noisy_audio) ** 2).item()
            input_distance_after = torch.mean((output_after - noisy_audio) ** 2).item()

            print("MSE output_after - output_before:", output_change)
            print("MSE checkpoint_output - noisy:", input_distance_before)
            print("MSE finetuned_output - noisy:", input_distance_after)

    denoised_np = tensor_audio_to_np(output_after)
    denoised_np = denoised_np[:len(audio_np)]

    del output_after
    del noisy_audio
    del noisy_batch
    del optimizer
    del physics
    del sure_loss_fn
    del demucs_model
    del local_demucs
    del trainable_params
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return denoised_np

#SGMSE
SGMSE_CKPT_PATH="../../../data/sgmse_voicebank.ckpt"
if not os.path.isfile(SGMSE_CKPT_PATH): import gdown; os.makedirs(os.path.dirname(SGMSE_CKPT_PATH),exist_ok=True); gdown.download(id="1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ", output=SGMSE_CKPT_PATH, quiet=False)

class SGMSEWaveformDenoiserForSURE(torch.nn.Module):
    def __init__(
        self,
        checkpoint_path,
        N=2,
        predictor="reverse_diffusion",
        corrector="ald",
        corrector_steps=1,
        snr=0.5,
        t_eps=0.03,
        sampler_seed=1234,
    ):
        super().__init__()

        self.score_model = ScoreModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        self.score_model.to(device)

        # SGMSE has a custom eval/train behavior with EMA weights.
        # Calling eval() once here copies EMA weights into the DNN.
        # After this, do NOT call score_model.train(), because that can restore
        # the non-EMA weights.
        self.score_model.eval()

        self.score_model.t_eps = float(t_eps)

        self.N = int(N)
        self.predictor = predictor
        self.corrector = corrector
        self.corrector_steps = int(corrector_steps)
        self.snr = float(snr)
        self.t_eps = float(t_eps)
        self.sampler_seed = sampler_seed

        if self.score_model.backbone == "ncsnpp_48k":
            self.target_sr = 48000
            self.pad_mode = "reflection"
        elif self.score_model.backbone == "ncsnpp_v2":
            self.target_sr = 16000
            self.pad_mode = "reflection"
        else:
            self.target_sr = 16000
            self.pad_mode = "zero_pad"

        if self.score_model.sde.__class__.__name__ != "OUVESDE":
            raise ValueError(
                "This SGMSE SURE wrapper currently supports OUVESDE checkpoints only. "
                f"Got {self.score_model.sde.__class__.__name__}."
            )

    def _prepare_spec(self, y_wave):
        """
        Convert waveform to the SGMSE complex spectrogram condition.

        Input
        -----
        y_wave:
            [B, 1, T]

        Output
        ------
        Y:
            SGMSE spectrogram condition, padded.
        norm_factors:
            [B, 1, 1]
        T:
            original waveform length
        """

        B, C, T = y_wave.shape

        if C != 1:
            raise ValueError("SGMSE SURE expects mono audio with shape [B, 1, T].")

        specs = []
        norm_factors = []

        for b in range(B):
            y_b = y_wave[b]  # [1, T]

            norm = y_b.abs().max().clamp_min(1e-8)
            y_b_norm = y_b / norm

            Y_b = self.score_model._forward_transform(
                self.score_model._stft(y_b_norm)
            )

            Y_b = torch.unsqueeze(Y_b, 0)
            Y_b = pad_spec(Y_b, mode=self.pad_mode)

            specs.append(Y_b)
            norm_factors.append(norm)

        Y = torch.cat(specs, dim=0)
        norm_factors = torch.stack(norm_factors).view(B, 1, 1)

        return Y, norm_factors, T

    def _differentiable_pc_sampler(self, Y):
        """
        Differentiable version of SGMSE PC sampling.

        The official SGMSE get_pc_sampler() uses torch.no_grad().
        For SURE, we cannot use no_grad here because DeepInv needs gradients.
        """

        sde = self.score_model.sde.copy()
        sde.N = self.N

        predictor_cls = PredictorRegistry.get_by_name(self.predictor)
        corrector_cls = CorrectorRegistry.get_by_name(self.corrector)

        predictor = predictor_cls(
            sde,
            self.score_model,
            probability_flow=False,
        )

        corrector = corrector_cls(
            sde,
            self.score_model,
            snr=self.snr,
            n_steps=self.corrector_steps,
        )

        # Make the stochastic sampler deterministic for a given forward call.
        # This is important because DeepInv SURE compares f(y) and f(y + tau*b).
        if self.sampler_seed is not None:
            torch.manual_seed(int(self.sampler_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.sampler_seed))

        xt = sde.prior_sampling(Y.shape, Y).to(Y.device)

        timesteps = torch.linspace(
            sde.T,
            self.t_eps,
            sde.N,
            device=Y.device,
        )

        for i in range(sde.N):
            t = timesteps[i]

            if i != len(timesteps) - 1:
                stepsize = t - timesteps[i + 1]
            else:
                stepsize = timesteps[-1]

            vec_t = torch.ones(Y.shape[0], device=Y.device) * t

            xt, xt_mean = corrector.update_fn(
                xt,
                Y,
                vec_t,
            )

            xt, xt_mean = predictor.update_fn(
                xt,
                Y,
                vec_t,
                stepsize,
            )

        return xt_mean

    def forward(self, y, physics=None, *args, **kwargs):
        """
        DeepInv-compatible forward.

        Input:
            y: [B, 1, T]

        Output:
            x_hat: [B, 1, T]
        """

        Y, norm_factors, T_orig = self._prepare_spec(y)

        sample = self._differentiable_pc_sampler(Y)

        outputs = []

        for b in range(y.shape[0]):
            x_hat_b = self.score_model.to_audio(
                sample[b].squeeze(),
                T_orig,
            )

            x_hat_b = x_hat_b * norm_factors[b].squeeze()
            x_hat_b = x_hat_b.view(1, -1)

            outputs.append(x_hat_b)

        x_hat = torch.stack(outputs, dim=0)

        return x_hat


def sgmse_sure_denoise(
    audio_np: np.ndarray,
    sigma: float,
    checkpoint_path: str,
    fs: int | None = None,
    steps: int = 1,
    lr: float = 1e-5,
    tau: float = 1e-3,
    mc_batch_size: int = 1,
    sgmse_N: int = 2,
    corrector: str = "ald",
    corrector_steps: int = 1,
    snr: float = 0.5,
    t_eps: float = 0.03,
    grad_clip: float = 1.0,
    debug: bool = False,
):
    """
    SGMSE + SURE fine-tuning.

    Each call:
        - loads a fresh SGMSE checkpoint
        - uses the full noisy audio
        - fine-tunes SGMSE with DeepInv SURE
        - returns a NumPy waveform

    Recommended first settings:
        steps=1
        sgmse_N=2
        mc_batch_size=1
        lr=1e-5

    Do not start with SGMSE N=30 here.
    Backpropagating through diffusion sampling is very expensive.
    """

    sgmse_model = SGMSEWaveformDenoiserForSURE(
        checkpoint_path=checkpoint_path,
        N=sgmse_N,
        predictor="reverse_diffusion",
        corrector=corrector,
        corrector_steps=corrector_steps,
        snr=snr,
        t_eps=t_eps,
        sampler_seed=1234,
    ).to(device)

    # Optional sample-rate sanity check.
    # Your current experiment already resamples to 16 kHz before denoising.
    if fs is not None:
        fs = int(fs)
        if fs != int(sgmse_model.target_sr):
            raise ValueError(
                f"SGMSE checkpoint expects {sgmse_model.target_sr} Hz, "
                f"but received fs={fs}. Resample before calling sgmse_sure_denoise."
            )

    # Do NOT call sgmse_model.train().
    # Do NOT call sgmse_model.score_model.train().
    # SGMSE train/eval controls EMA restoration/copying.
    for param in sgmse_model.parameters():
        param.requires_grad = True

    trainable_params = [
        p for p in sgmse_model.parameters()
        if p.requires_grad
    ]

    if debug:
        n_trainable = sum(p.numel() for p in trainable_params)
        print("Trainable SGMSE parameters:", n_trainable)
        print("SGMSE target_sr:", sgmse_model.target_sr)
        print("SGMSE pad_mode:", sgmse_model.pad_mode)

    noisy_audio = np_audio_to_tensor(audio_np)  # [1, 1, T]

    if noisy_audio.shape[1] != 1:
        raise ValueError("SGMSE SURE currently expects mono audio.")

    noisy_batch = noisy_audio.repeat(mc_batch_size, 1, 1)

    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(
            sigma=float(sigma)
        )
    )

    sure_loss_fn = dinv.loss.SureGaussianLoss(
        sigma=float(sigma),
        tau=float(tau),
    )

    # More conservative than Demucs.
    effective_lr = float(lr) / (1.0 + 10.0 * float(sigma) ** 2)

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=effective_lr,
    )

    best_loss = float("inf")
    best_state = None
    previous_sure = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        x_net = sgmse_model(noisy_batch)

        loss_per_item = sure_loss_fn(
            x_net=x_net,
            y=noisy_batch,
            physics=physics,
            model=sgmse_model,
        )

        loss = loss_per_item.mean()

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(grad_clip),
            )

        optimizer.step()

        current_sure = float(loss.detach().cpu())
        for group in optimizer.param_groups:
            group["lr"] = effective_lr * 1
        '''
        if current_sure < 1e-1:
            for group in optimizer.param_groups:
                group["lr"] = effective_lr * 0.7
                
        if current_sure < 1e-2:
            for group in optimizer.param_groups:
                group["lr"] = effective_lr * 0.2

        if current_sure < 6e-3:
            for group in optimizer.param_groups:
                group["lr"] = effective_lr * 0.08
'''
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"SGMSE SURE step {step + 1}/{steps} | "
            f"sigma={sigma:.3e} | "
            f"lr={current_lr:.3e} | "
            f"SURE={current_sure:.6e} | "
            f"min={loss_per_item.min().item():.6e} | "
            f"max={loss_per_item.max().item():.6e}"
        )

        if np.isfinite(current_sure) and current_sure < best_loss:
            best_loss = current_sure
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in sgmse_model.state_dict().items()
            }

        if not np.isfinite(current_sure):
            print("Stopping SGMSE SURE: SURE became NaN/inf")
            break

        if previous_sure is not None and current_sure > 5.0 * previous_sure:
            print("Stopping SGMSE SURE: SURE exploded")
            break

        previous_sure = current_sure
        del x_net
        del loss
        del loss_per_item

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if best_state is not None:
        sgmse_model.load_state_dict(best_state)

    with torch.no_grad():
        output_after = sgmse_model(noisy_audio)

    denoised_np = tensor_audio_to_np(output_after)
    denoised_np = denoised_np[:len(audio_np)]

    del output_after
    del noisy_audio
    del noisy_batch
    del optimizer
    del physics
    del sure_loss_fn
    del sgmse_model
    del trainable_params

    if best_state is not None:
        del best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return denoised_np


#UNSURE
class AudioUnsureDenoiser(nn.Module):
    """
    Small trainable 1D denoiser used only for UNSURE sigma estimation.

    DeepInv SURE code usually uses image-like tensors [B, C, H, W].
    For mono audio, we use [B, 1, 1, T], squeeze H=1, apply Conv1d,
    then restore [B, 1, 1, T].
    """

    def __init__(self, hidden_channels=32, kernel_size=15):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size, padding=padding),
        )

    def forward(self, y, *args, **kwargs):
        if y.ndim == 4:
            # [B, 1, 1, T] -> [B, 1, T]
            y_1d = y[:, :, 0, :]
            x_1d = self.net(y_1d)
            # [B, 1, T] -> [B, 1, 1, T]
            return x_1d[:, :, None, :]

        if y.ndim == 3:
            # [B, 1, T]
            return self.net(y)

        raise ValueError("AudioUnsureDenoiser expects [B,1,T] or [B,1,1,T].")


def robust_sigma_init_from_differences(noisy_sound):
    """
    Robust initialization of AWGN sigma from first differences.

    This is not the final estimate. It only initializes UNSURE.
    For white noise n, diff(n) has std sqrt(2) * sigma.
    """

    y = np.asarray(noisy_sound, dtype=np.float32).squeeze()

    if y.ndim != 1:
        raise ValueError("noisy_sound must be a mono 1D signal.")

    if y.size < 2:
        return 1e-2

    diff = np.diff(y)
    mad = np.median(np.abs(diff - np.median(diff)))

    sigma_init = mad / (0.67448975 * np.sqrt(2.0))

    if not np.isfinite(sigma_init) or sigma_init <= 0:
        sigma_init = 1e-2

    return float(np.clip(sigma_init, 1e-6, 1.0))


def sample_audio_unsure_patches(y, batch_size, patch_size):
    """
    Random patch sampler for one audio signal.

    Parameters
    ----------
    y:
        Tensor with shape [1, 1, 1, T]
    batch_size:
        Number of random patches.
    patch_size:
        Patch length in samples.

    Returns
    -------
    patches:
        Tensor with shape [batch_size, 1, 1, patch_size]
    """

    length = y.shape[-1]
    patch_size = min(int(patch_size), int(length))

    if length == patch_size:
        return y.repeat(batch_size, 1, 1, 1)

    starts = torch.randint(
        low=0,
        high=length - patch_size + 1,
        size=(batch_size,),
        device=y.device,
    )

    patches = []
    for start in starts:
        start = int(start.item())
        patches.append(y[:, :, :, start:start + patch_size])

    return torch.cat(patches, dim=0)


def estimate_sigma_unsure_audio(
    noisy_sound,
    sigma_init=None,
    steps=150,
    batch_size=8,
    patch_size=8192,
    lr=1e-3,
    tau=1e-3,
    unsure_step_size=1e-4,
    unsure_momentum=0.9,
    hidden_channels=32,
    kernel_size=15,
    device=None,
    verbose=False,
):
    """
    Estimate scalar white Gaussian noise sigma from one noisy mono audio extract
    using DeepInv's UNSURE version of SureGaussianLoss.

    Parameters
    ----------
    noisy_sound:
        1D NumPy array.
    sigma_init:
        Initial sigma. If None, uses robust_sigma_init_from_differences.
    steps:
        Number of UNSURE optimization steps.
    batch_size:
        Number of random patches per step.
    patch_size:
        Patch size in audio samples.
    lr:
        Adam learning rate for the small denoiser.
    tau:
        Monte Carlo SURE perturbation size.
    unsure_step_size:
        Gradient-ascent step size for the UNSURE noise level.
    unsure_momentum:
        Momentum for the UNSURE noise-level update.
    hidden_channels:
        Width of the small Conv1d denoiser.
    kernel_size:
        Kernel size of the Conv1d denoiser.
    device:
        Optional torch device.
    verbose:
        If True, prints progress.

    Returns
    -------
    sigma_hat:
        Estimated sigma in the same amplitude scale as noisy_sound.
    """

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    y_np = np.asarray(noisy_sound, dtype=np.float32).squeeze()

    if y_np.ndim != 1:
        raise ValueError("noisy_sound must be a mono 1D NumPy array.")

    if sigma_init is None:
        sigma_init = robust_sigma_init_from_differences(y_np)

    sigma_init = float(sigma_init)

    if not np.isfinite(sigma_init) or sigma_init <= 0:
        sigma_init = 1e-2

    y = torch.from_numpy(y_np).float().view(1, 1, 1, -1).to(device)

    physics = dinv.physics.Denoising()

    model = AudioUnsureDenoiser(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    ).to(device)

    unsure_loss = dinv.loss.SureGaussianLoss(
        sigma=sigma_init,
        tau=float(tau),
        unsure=True,
        step_size=float(unsure_step_size),
        momentum=float(unsure_momentum),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    model.train()

    for step in range(int(steps)):
        y_batch = sample_audio_unsure_patches(
            y,
            batch_size=int(batch_size),
            patch_size=int(patch_size),
        )

        optimizer.zero_grad(set_to_none=True)

        x_net = model(y_batch)

        loss = unsure_loss(
            y=y_batch,
            x_net=x_net,
            physics=physics,
            model=model,
        ).mean()

        loss.backward()
        optimizer.step()

        if verbose and (step % 25 == 0 or step == steps - 1):
            sigma_now = torch.sqrt(unsure_loss.sigma2.detach().clamp_min(0.0)).item()
            print(
                f"UNSURE step {step + 1}/{steps} | "
                f"loss={loss.item():.4e} | sigma={sigma_now:.4e}"
            )

    sigma_hat = torch.sqrt(unsure_loss.sigma2.detach().clamp_min(0.0)).item()

    if not np.isfinite(sigma_hat) or sigma_hat <= 0:
        sigma_hat = sigma_init

    return float(sigma_hat)







#------------------------------------#
#-----------  ARCHIVES --------------#
#------------------------------------#


'''
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


def deepinv_sure_demucs_drywet_search_cached(
    y_np,
    drywet_values,
    sigma,
    fs,
    chunk_size=65536,
    device=None,
    tau=0.01,
    n_sure_repeats=1,
    param_name="drywet",
):
    """
    SURE search for DEMUCS dry/wet without rerunning DEMUCS for every dry/wet.

    Instead of doing:

        for drywet in drywet_values:
            run DEMUCS
            compute SURE

    this function does:

        run DEMUCS once on y
        run DEMUCS once per SURE perturbation
        evaluate all dry/wet values cheaply by linear interpolation

    Denoiser form:
        A_drywet(y) = drywet * DEMUCS(y) + (1 - drywet) * y

    This keeps the finite-difference SURE estimate correct for the dry/wet
    family, while avoiding repeated DEMUCS calls for every dry/wet value.
    """

    if device is None:
        device = dinv.utils.get_device()

    y_np = np.asarray(y_np, dtype=np.float32)

    if y_np.ndim != 1:
        raise ValueError("y_np must be a mono 1D NumPy array.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    drywet_values = np.asarray(drywet_values, dtype=float)

    if np.any(drywet_values < 0) or np.any(drywet_values > 1):
        raise ValueError("drywet_values should be between 0 and 1.")

    original_len = len(y_np)

    pad_len = (-original_len) % chunk_size

    if pad_len > 0:
        y_padded = np.pad(y_np, (0, pad_len), mode="reflect")
    else:
        y_padded = y_np

    chunks = y_padded.reshape(-1, chunk_size)

    # DeepInv-style tensor shape for 1D audio: [chunks, 1, 1, chunk_size]
    y_tensor = torch.from_numpy(chunks[:, None, None, :]).to(device)

    # Number of scalar samples per chunk, used for normalized SURE.
    n_per_chunk = y_tensor[0].numel()

    # ------------------------------------------------------------
    # 1. Run DEMUCS once on the clean noisy input y.
    # ------------------------------------------------------------
    demucs_y = _demucs_forward_tensor_chunks(y_tensor, fs)

    sure_accum = torch.zeros(
        len(drywet_values),
        device=device,
        dtype=y_tensor.dtype,
    )

    # ------------------------------------------------------------
    # 2. SURE Monte Carlo loop.
    #    DEMUCS is run once per perturbation, not once per dry/wet.
    # ------------------------------------------------------------
    for _ in range(n_sure_repeats):
        b = torch.randn_like(y_tensor)
        y_perturbed = y_tensor + tau * b

        demucs_y_perturbed = _demucs_forward_tensor_chunks(y_perturbed, fs)

        for i, drywet in enumerate(drywet_values):
            drywet = float(drywet)

            # A(y)
            x_net = drywet * demucs_y + (1.0 - drywet) * y_tensor

            # A(y + tau b)
            x_net_perturbed = (
                drywet * demucs_y_perturbed
                + (1.0 - drywet) * y_perturbed
            )

            residual_term = (x_net - y_tensor).pow(2).flatten(1).sum(dim=1)
            residual_term = residual_term / n_per_chunk

            divergence_term = (
                b * (x_net_perturbed - x_net) / tau
            ).flatten(1).sum(dim=1)
            divergence_term = divergence_term / n_per_chunk

            sure_per_chunk = (
                residual_term
                - float(sigma) ** 2
                + 2.0 * float(sigma) ** 2 * divergence_term
            )

            sure_accum[i] += sure_per_chunk.mean()

    sure_values = (sure_accum / n_sure_repeats).detach().cpu().numpy()

    best_idx = int(np.argmin(sure_values))
    best_drywet = float(drywet_values[best_idx])

    print("SURE best index:", best_idx, "/", len(drywet_values) - 1)
    print(f"SURE best {param_name}:", best_drywet)
    print("SURE first value:", sure_values[0])
    print("SURE best value:", sure_values[best_idx])
    print("SURE last value:", sure_values[-1])

    if best_idx == len(drywet_values) - 1:
        print(f"WARNING: SURE selected the maximum tested {param_name}.")

    if best_idx == 0:
        print(f"WARNING: SURE selected the minimum tested {param_name}.")

    # ------------------------------------------------------------
    # 3. Build best denoised signal without rerunning DEMUCS.
    # ------------------------------------------------------------
    with torch.no_grad():
        x_best = best_drywet * demucs_y + (1.0 - best_drywet) * y_tensor

    denoised_chunks = x_best.detach().cpu().numpy()[:, 0, 0, :]
    best_denoised = denoised_chunks.reshape(-1)[:original_len]

    return best_denoised, best_drywet, sure_values