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
from tools import find_oracle_threshold, denoise_spectral_sub

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_soustraction_spectrale"
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


#Parcours la liste des iSNR de params.txt
for isnr in params['isnr']:
    #Bruitage
    noisy_sound,sigma=noise.add_white_noise(sound,samplerate,isnr)
    
    #Recherche seuil oracle
    oracle_threshold,figure,thresholds,oSNR_list=find_oracle_threshold(sound,noisy_sound,sigma,samplerate,denoise_spectral_sub,True)

    #Debruitage par seuil oracle
    _,denoised_sound=denoise_spectral_sub(noisy_sound, sigma, oracle_threshold, samplerate)
    
    #Export des resultats
    if len(CSV_export)==0:
        thresholdss=thresholds.tolist()
        thresholdss.insert(0,'Threshold')
        CSV_export.append(thresholdss)
    oSNR_list.insert(0,isnr)
    CSV_export.append(oSNR_list)
    figure.savefig(path+"/seuil_oracle_isnr_"+str(isnr)+".jpg", dpi=300, bbox_inches="tight")
    if params['export_audio']:
        sf.write(path+'/noisy_sound_isnr_'+str(isnr)+'.wav', noisy_sound, samplerate)
        sf.write(path+'/denoised_sound_isnr_'+str(isnr)+'.wav', denoised_sound, samplerate)
    #Export du CSV
with open(path+'/SNR Soustraction Spectrale.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(CSV_export)
    
'''CSV Export format: Column 1 --> iSNR values
line 1 --> Thresholds
line n --> oSNR as function of threshold for iSNR
'''