#Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display
from scipy.signal import stft, istft
import sys
import csv
from datetime import datetime
import os

sys.path.append("../../utils")
import noise
from tools import find_oracle_threshold, denoise_spectral_sub, deepinv_sure_spectral_sub_threshold_search
from metrics import SNR

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_SURE"
os.mkdir(path)

CSV_export=[]

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

sigmas=np.logspace(-4,0,10)
thresholds=np.linspace(0,1,100)

for isnr in params['isnr']:
    SURE_oSNR=[]
    #Bruitage
    noisy_sound,sig=noise.add_white_noise(sound,samplerate,isnr)
    for sigma in sigmas:
        # Recherche seuil SURE avec la vraie fonction denoise_spectral_sub
        denoised_sure, sure_threshold, sure_values = deepinv_sure_spectral_sub_threshold_search(
            y_np=noisy_sound,
            thresholds=thresholds,
            sigma=sigma,
            fs=samplerate,
            denoise_func=denoise_spectral_sub,
            chunk_size=16384,
            tau=0.01,
            n_sure_repeats=1,
        )
        SURE_oSNR.append(SNR(sound,denoised_sure))
    
    plt.semilogx(sigmas,SURE_oSNR,label="SURE")
    plt.xlabel(r"$\sigma$")
    plt.ylabel('Mean oSNR')
    plt.title('oSNR - SURE')
    plt.grid(True,which='both')
    plt.legend()
    plt.savefig(path+"/oSNR_SURE_iSNR_"+str(isnr)+".jpg", dpi=300, bbox_inches="tight")
    plt.show()