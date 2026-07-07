#Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import csv
from datetime import datetime
import os
import torch
from denoiser.dsp import convert_audio

sys.path.append("../../utils")
import noise
from metrics import SNR,signal_power
from tools import find_oracle_threshold,denoise_spectral_sub,demucs_denoise,deepinv_sure_param_search,SpectralSubNumpyWrapper,deepinv_sure_spectral_sub_threshold_search,deepinv_sure_demucs_drywet_search_cached,deepinv_sure_spectral_sub_threshold_search,deepinv_sure_spectral_sub_threshold_search,deepinv_sure_spectral_sub_threshold_search_torch, demucs_sure_denoise

#Creation du dossier de l'experience
epath="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_oSNR_sigma_sure_soustraction_spectrale_demucs"
os.mkdir(epath)

CSV_export=[]

#Requete des donnees params.txt
parameters=open('params_multiple.txt')
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
            else:
                try:
                    val=float(val)
                except:
                    val=val
        params[key.strip()]=val

if type(params['isnr'])!=list:
    params['isnr']=[params['isnr']]

params['sigma']=np.logspace(-3,0,5)
params['drywet']=np.linspace(0,1,100)

#Import du dossier sonore
sound_folder='../../../data/'+params['pure_sound_folder']
sound_files=os.listdir(sound_folder)

sound_files_clean=[]
for folder in sorted(sound_files):
    wav_path = os.path.join(sound_folder, folder, params["sound_type"] + ".wav")
    if os.path.isfile(wav_path):
        sound_files_clean.append(os.path.join(folder, params["sound_type"] + ".wav"))
    if len(sound_files_clean) >= params["track_samples"]:
        break

#Initialisation des resultats
all_oSNR={}
all_oSNR_demucs={}
all_sure_thresholds={}
all_sigma_true={}

#Parcours la liste des iSNR de params.txt
for isnr in params['isnr']:
    sound_number=0
    all_oSNR[isnr]=[]
    all_oSNR_demucs[isnr]=[]
    all_sure_thresholds[isnr]=[]
    all_sigma_true[isnr]=[]

    #Parcours les fichiers sonores
    for sound_file in sound_files_clean:
        sound_number+=1
        #Import du fichier sonore
        info=sf.info(sound_folder+'/'+sound_file)
        samplerate=info.samplerate

        duration_extract=10
        extract_length=duration_extract*samplerate

        if info.frames>extract_length:
            start=np.random.randint(0,info.frames-extract_length)
            sound,samplerate=sf.read(sound_folder+'/'+sound_file,start=start,frames=extract_length)
        else:
            sound,samplerate=sf.read(sound_folder+'/'+sound_file)
        if len(sound.shape)>1:
            sound=np.mean(sound,axis=1)
        sound=convert_audio(torch.from_numpy(sound).float()[None,:],samplerate,16000,1)[0].detach().cpu().numpy()
        samplerate=16000

        #Bruitage
        noisy_sound,sigma_true=noise.add_white_noise(sound,samplerate,isnr)

        if sigma_true<10**-2:
            continue
        
        #Creation de la liste des seuils a tester dans SURE
        oracle_threshold,figure,thresholds,oSNR_threshold_list=find_oracle_threshold(
            sound,
            noisy_sound,
            sigma_true,
            samplerate,
            denoise_spectral_sub,
            True
        )
        plt.close(figure)

        oSNR_sigma_list=[]
        oSNR_sigma_list_demucs=[]
        sure_threshold_list=[]

        #Parcours des sigma donnes a SURE
        for sigma in params['sigma']:

            #Recherche du seuil SURE avec ce sigma
            denoised_sure, sure_threshold, sure_values = deepinv_sure_spectral_sub_threshold_search_torch(
                y_np=noisy_sound,
                thresholds=thresholds,
                sigma=sigma,
                fs=samplerate,
                chunk_size=65536,
                tau=0.01,
                n_sure_repeats=1,
            )

            #Calcul oSNR
            _,denoised_sure_specsub=denoise_spectral_sub(noisy_sound,sigma,sure_threshold,samplerate)
            osnr=SNR(sound,denoised_sure_specsub)

            oSNR_sigma_list.append(osnr)
            sure_threshold_list.append(sure_threshold)

            #Recherche du dry/wet SURE avec ce sigma pour DEMUCS
            denoised_sure_demucs = demucs_sure_denoise(
                noisy_sound,sigma,50,1e-5
            )

            #Calcul oSNR DEMUCS
            osnr_demucs=SNR(sound,denoised_sure_demucs)

            oSNR_sigma_list_demucs.append(osnr_demucs)

        #Graphe individuel oSNR en fonction de sigma
        sigma_array=np.array(params['sigma'])
        oSNR_sigma_array=np.array(oSNR_sigma_list)
        oSNR_sigma_array_demucs=np.array(oSNR_sigma_list_demucs)

        mask=sigma_array>0

        figure=plt.figure(figsize=(8,5))

        plt.semilogx(
            sigma_array[mask],
            oSNR_sigma_array[mask],
            label='Soustraction spectrale, iSNR = '+str(isnr)+' dB'
        )

        plt.semilogx(
            sigma_array[mask],
            oSNR_sigma_array_demucs[mask],
            label='DEMUCS, iSNR = '+str(isnr)+' dB'
        )

        plt.axvline(
            sigma_true,
            color='black',
            linestyle='--',
            linewidth=2,
            label='True sigma'
        )

        plt.xlabel('Sigma given to SURE')
        plt.ylabel('oSNR')
        plt.title('oSNR as function of sigma given to SURE - '+str(sound_number))
        plt.grid(True,which='both')
        plt.legend()

        #safe_sound_file=os.path.splitext(sound_file)[0].replace(' ','_')

        figure.savefig(
            epath+"/individual_oSNR_sigma_"+str(sound_number)+"_iSNR_"+str(isnr)+".jpg",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)

        all_oSNR[isnr].append(oSNR_sigma_list)
        all_oSNR_demucs[isnr].append(oSNR_sigma_list_demucs)
        all_sure_thresholds[isnr].append(sure_threshold_list)
        all_sigma_true[isnr].append(sigma_true)

#Nombre d'extraits
N=len(sound_files_clean)

#Creation de la figure finale
plt.figure(figsize=(8,5))

#Export du CSV
CSV_export.append([
    'iSNR',
    'Sigma given to SURE',
    'Mean oSNR SpecSub',
    'Standard deviation oSNR SpecSub',
    'Uncertainty oSNR SpecSub',
    'Mean SURE threshold',
    'Std SURE threshold',
    'Uncertainty SURE threshold',
    'Mean oSNR DEMUCS',
    'Standard deviation oSNR DEMUCS',
    'Uncertainty oSNR DEMUCS',
    'Mean SURE drywet',
    'Std SURE drywet',
    'Uncertainty SURE drywet',
    'Mean true sigma',
    'Std true sigma',
    'Uncertainty true sigma',
    'Number of extracts'
])

for isnr in params['isnr']:

    oSNR_array=np.array(all_oSNR[isnr])
    oSNR_array_demucs=np.array(all_oSNR_demucs[isnr])
    sure_thresholds_array=np.array(all_sure_thresholds[isnr])
    sigma_true_array=np.array(all_sigma_true[isnr])
    sigma_array=np.array(params['sigma'])

    mean_oSNR=np.mean(oSNR_array,axis=0)

    if N>1:
        std_oSNR=np.std(oSNR_array,axis=0,ddof=1)
    else:
        std_oSNR=np.zeros(len(mean_oSNR))

    uncertainty=std_oSNR/np.sqrt(N)

    mean_oSNR_demucs=np.mean(oSNR_array_demucs,axis=0)

    if N>1:
        std_oSNR_demucs=np.std(oSNR_array_demucs,axis=0,ddof=1)
    else:
        std_oSNR_demucs=np.zeros(len(mean_oSNR_demucs))

    uncertainty_demucs=std_oSNR_demucs/np.sqrt(N)

    mean_sure_threshold=np.mean(sure_thresholds_array,axis=0)

    if N>1:
        std_sure_threshold=np.std(sure_thresholds_array,axis=0,ddof=1)
    else:
        std_sure_threshold=np.zeros(len(mean_sure_threshold))

    sure_uncertainty=std_sure_threshold/np.sqrt(N)



    mean_sigma_true=np.mean(sigma_true_array)

    if N>1:
        std_sigma_true=np.std(sigma_true_array,ddof=1)
    else:
        std_sigma_true=0

    sigma_true_uncertainty=std_sigma_true/np.sqrt(N)

    mask=sigma_array>0

    plt.semilogx(
        sigma_array[mask],
        mean_oSNR[mask],
        label='Soustraction spectrale, iSNR = '+str(isnr)+' dB'
    )

    plt.fill_between(
        sigma_array[mask],
        mean_oSNR[mask]-uncertainty[mask],
        mean_oSNR[mask]+uncertainty[mask],
        alpha=0.2
    )

    plt.semilogx(
        sigma_array[mask],
        mean_oSNR_demucs[mask],
        label='DEMUCS, iSNR = '+str(isnr)+' dB'
    )

    plt.fill_between(
        sigma_array[mask],
        mean_oSNR_demucs[mask]-uncertainty_demucs[mask],
        mean_oSNR_demucs[mask]+uncertainty_demucs[mask],
        alpha=0.2
    )

    plt.axvline(
        mean_sigma_true,
        color='black',
        linestyle='--',
        linewidth=2,
        label='Mean true sigma, iSNR = '+str(isnr)+' dB'
    )

    plt.axvspan(
        max(mean_sigma_true-sigma_true_uncertainty,np.min(sigma_array[mask])),
        mean_sigma_true+sigma_true_uncertainty,
        color='black',
        alpha=0.15
    )

    for i in range(len(sigma_array)):
        CSV_export.append([
            isnr,
            sigma_array[i],
            mean_oSNR[i],
            std_oSNR[i],
            uncertainty[i],
            mean_sure_threshold[i],
            std_sure_threshold[i],
            sure_uncertainty[i],
            mean_oSNR_demucs[i],
            std_oSNR_demucs[i],
            uncertainty_demucs[i],
            mean_sigma_true,
            std_sigma_true,
            sigma_true_uncertainty,
            N
        ])

plt.xlabel('Sigma estimation for SURE')
plt.ylabel('oSNR (dB)')
plt.title('SURE optimal oSNR for various denoising methods')
plt.grid(True,which='both')
plt.legend()
plt.savefig(epath+"/mean_oSNR_as_function_of_sigma_sure.jpg",dpi=300,bbox_inches="tight")
plt.show()

#Export du CSV du graphe
with open(epath+'/Mean oSNR as function of sigma SURE.csv','w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerows(CSV_export)