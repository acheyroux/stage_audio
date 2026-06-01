#Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft, istft
import sys
from denoiser.dsp import convert_audio
import torch
from datetime import datetime
import os
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec

sys.path.append("../../utils")
import noise
from metrics import SNR

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_sgmse"
os.mkdir(path)
ckpt_path="../../../Data/train_wsj0_2cta4cov_epoch=159.ckpt"

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

#Import du fichier sonore et modele SGMSE+
sound,samplerate=sf.read('../../../data/'+params['pure_sound'])
model=ScoreModel.load_from_checkpoint(ckpt_path, map_location=params['sgmse_device'])
model.eval()
model.to(params['sgmse_device'])
model_sample_rate=16000

sound=sound.astype(np.float32)
son_t=convert_audio(torch.from_numpy(sound).float()[None, :],44100,16000,1)
sound=son_t[0].detach().cpu().numpy()
#Parcours la liste des iSNR de params.txt
SNR_results=''
for isnr in params['isnr']:
    #Bruitage
    son_b,sigma=noise.add_white_noise(sound,model_sample_rate,isnr)
    son_bt=torch.from_numpy(son_b).float()[None, :]
    #Debruitage SGMSE+
    Y=model._forward_transform(model._stft(son_bt))
    Y=Y[None]
    Y=pad_spec(Y, mode="reflection")
    sampler = model.get_pc_sampler("reverse_diffusion","ald",Y.to(params['sgmse_device']),N=params['sgmse_layers'],corrector_steps=1,snr=.5,)
    with torch.no_grad():
        sample, _ = sampler()
    son_dt=model.to_audio(sample.squeeze(), son_t.shape[-1])
    son_d=son_dt[0, 0].detach().cpu().numpy()
    osnr=SNR(sound,son_d)
    
    #Export des resultats
    #figure.savefig(path+"".jpg", dpi=300, bbox_inches="tight")  --ignore
    SNR_results+="sigma="+str(sigma)+"\tiSNR="+str(isnr)+"\tSNR="+str(osnr)+"\n"
    if params['export_audio']:
        sf.write(path+'/noisy_sound_isnr_'+str(isnr)+'.wav', son_b, model_sample_rate)
        sf.write(path+'/SGMSE_denoised_sound_isnr_'+str(isnr)+'.wav', son_d, model_sample_rate)
    
#Creation du fichier SNR.txt
with open(path+'/SNR SGMSE+.txt','w') as d:
   d.write(SNR_results)

#Copiage des parametres utilises
with open("params.txt", 'r') as file:
    content = file.read()
with open(path+"/params.txt", 'w') as otherfile:
    otherfile.write(content)