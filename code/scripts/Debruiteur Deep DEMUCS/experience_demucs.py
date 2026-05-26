#Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft, istft
import sys
from denoiser import pretrained #DEMUCS
from denoiser.dsp import convert_audio
import torch
from datetime import datetime
import os

sys.path.append("../../utils")
import noise
from metrics import SNR

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_demucs"
os.mkdir(path)

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

#Import du fichier sonore et modele DEMUCS
sound,samplerate=sf.read('../../../data/'+params['pure_sound'])
model=getattr(pretrained, params['demucs_pretrained_model'])()

#Parcours la liste des iSNR de params.txt
SNR_results=''
for isnr in params['isnr']:
    #Bruitage
    son_b,sigma=noise.add_white_noise(sound,samplerate,isnr)
    
    #Debruitage DEMUCS
    model_sample_rate=model.sample_rate
    son_bt=convert_audio(torch.from_numpy(son_b).float()[None, :],samplerate,model_sample_rate,1)
    son_dt=model(son_bt[None])
    son_d=son_dt[0, 0].detach().cpu().numpy()
    osnr=SNR(sound,son_d)
    
    #Export des resultats
    #figure.savefig(path+"".jpg", dpi=300, bbox_inches="tight")  --ignore
    SNR_results+="iSNR="+str(isnr)+"\tSNR="+str(osnr)+"\n"
    if params['export_audio']:
        sf.write(path+'/noisy_sound_isnr_'+str(isnr)+'.wav', son_b, samplerate)
        sf.write(path+'/DEMUCS_denoised_sound_isnr_'+str(isnr)+'.wav', son_d, model_sample_rate)
    
#Creation du fichier SNR.txt
with open(path+'/SNR.txt','w') as d:
   d.write(SNR_results)

#Copiage des parametres utilises
with open("params.txt", 'r') as file:
    content = file.read()
with open(path+"/params.txt", 'w') as otherfile:
    otherfile.write(content)