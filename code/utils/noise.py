import numpy as np
import matplotlib.pyplot as plt
from metrics import signal_power

def white_noise(N: int, fs: float, mu: float, sigma: float,seed=0,show=False):
    '''
    Cree un signal 'bruit blanc'
    --- In ---
    N : longueur du signal (int)
    fs : frequence d'echantillonage (float)
    mu : moyenne du bruit (float)
    sigma : ecart-type du bruit (float)
    seed : graine np.random (int)
    --- Out ---
    noise : signal bruite (ndarray)
    '''
    np.random.seed(seed)
    noise=np.random.normal(mu,sigma,N)
    if show:
        signal(noise,fs,True)
    return noise

def add_white_noise(x, fs: float, target_SNR: float, seed=0, show=False):
    '''
    Ajoute du bruit blanc au signal jusqu'a obtenir la DSP totale recherchee.
    --- In ---
    x : signal (ndarray)
    fs : frequence d'echantillonage (float)
    target_SNR : SNR recherchee en sortie en dB (float)
    show : trace le signal en sortie (bool)
    --- Out ---
    x_b : signal bruite (ndarray)
    target_sigma : ecart type du bruit (float)
    '''
    target_sigma=(signal_power(x)*10**-(target_SNR/10))**.5
    x_b=x+white_noise(len(x),fs,0,target_sigma,seed,False)
    if show:
        signal(x_b,fs,True)
    return x_b,target_sigma