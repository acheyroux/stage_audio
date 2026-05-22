#Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display
from scipy.signal import stft, istft
import sys

sys.path.append("../../utils")
import noise
from tools import find_oracle_threshold, denoise_spectral_sub


#Requete des donnees params.txt
parameters=open('params.txt')
params={}
for full_line in parameters:
    line=full_line.strip()
    if not line[0]=='#':
        key,value=line.split('=',1)
        val=value.strip().split(',')
        if len(val)!=1:
            for i in range(len(val)):
                val[i]=float(val[i].strip())
        else:
            val=val[0].strip()
            if val.isdigit():
                val=int(val)
            elif val=='True':
                val=True
            elif val=='False':
                val=False
        params[key.strip()]=val

#Import du fichier sonore
sound,samplerate=sf.read('../../../data/'+params['pure_sound'])

#Parcours la liste des iSNR de params.txt
for isnr in params['isnr']:
    #Bruitage
    noisy_sound,sigma=noise.add_white_noise(sound,samplerate,isnr)
    
    #Recherche seuil oracle
    oracle_threshold,figure=find_oracle_threshold(sound,noisy_sound,sigma,samplerate,denoise_spectral_sub,True)

    #Debruitage par seuil oracle
    _,denoised_sound=denoise_spectral_sub(noisy_sound, sigma, oracle_threshold, samplerate)

    #Export des resultats
    figure.savefig("../../../results/exp_20260521_soustraction_spectrale_seuil_oracle/seuil_oracle_isnr_"+str(isnr)+".jpg", dpi=300, bbox_inches="tight")
    if params['export_audio']:
        sf.write('../../../results/exp_20260521_soustraction_spectrale_seuil_oracle/noisy_sound_isnr_'+str(isnr)+'.wav', noisy_sound, samplerate)
        sf.write('../../../results/exp_20260521_soustraction_spectrale_seuil_oracle/denoised_sound_isnr_'+str(isnr)+'.wav', denoised_sound, samplerate)