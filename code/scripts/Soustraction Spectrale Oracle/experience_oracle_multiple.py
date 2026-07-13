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
from tools import find_oracle_threshold, denoise_spectral_sub, deepinv_sure_spectral_sub_threshold_search_torch

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_multiple_soustraction_spectrale"
os.mkdir(path)

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
        params[key.strip()]=val
if type(params['isnr'])!=list:
    params['isnr']=[params['isnr']]

#Import du dossier sonore
sound_folder='../../../data/'+params['pure_sound_folder']
sound_files=os.listdir(sound_folder)

sound_files_clean=[]
for file in sound_files:
    if file.endswith('.wav'):
        sound_files_clean.append(file)

#Initialisation des resultats
all_oSNR={}
all_sure_thresholds = {}

#Parcours la liste des iSNR de params.txt
for isnr in params['isnr']:

    all_oSNR[isnr]=[]
    all_sure_thresholds[isnr] = []
    
    #Parcours les fichiers sonores
    for sound_file in sound_files_clean:

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

        #Bruitage
        noisy_sound,sigma=noise.add_white_noise(sound,samplerate,isnr)

        # Recherche seuil oracle
        oracle_threshold, figure, thresholds, oSNR_list = find_oracle_threshold(
            sound,
            noisy_sound,
            sigma,
            samplerate,
            denoise_spectral_sub,
            True
        )
        
        # Recherche seuil SURE avec la vraie fonction denoise_spectral_sub
        denoised_sure, sure_threshold, sure_values = deepinv_sure_spectral_sub_threshold_search_torch(
            y_np=noisy_sound,
            thresholds=thresholds,
            sigma=sigma,
            fs=samplerate,
            chunk_size=16384,
            tau=0.01,
            n_sure_repeats=1,
        )

        # Ajout du seuil SURE sur le graphe individuel oracle
        ax = figure.axes[0]

        ax.axvline(
            sure_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label='SURE threshold'
        )

        ax.legend()

        safe_sound_file = os.path.splitext(sound_file)[0].replace(' ', '_')

        figure.savefig(
            path + "/oracle_graph_" + safe_sound_file + "_iSNR_" + str(isnr) + ".jpg",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)
        
        # Debruitage par seuil oracle
        _, denoised_sound = denoise_spectral_sub(
            noisy_sound,
            sigma,
            oracle_threshold,
            samplerate
        )
        
        all_oSNR[isnr].append(oSNR_list)
        all_sure_thresholds[isnr].append(sure_threshold)

#Nombre d'extraits
N=len(sound_files_clean)

#Creation de la figure finale
plt.figure(figsize=(8,5))

#Export du CSV
CSV_export.append([
    'iSNR',
    'Threshold',
    'Mean oSNR',
    'Standard deviation',
    'Uncertainty',
    'Mean SURE threshold',
    'Std SURE threshold',
    'Uncertainty SURE threshold'
])

for isnr in params['isnr']:

    oSNR_array=np.array(all_oSNR[isnr])

    mean_oSNR=np.mean(oSNR_array,axis=0)

    if N>1:
        std_oSNR=np.std(oSNR_array,axis=0,ddof=1)
    else:
        std_oSNR=np.zeros(len(mean_oSNR))

    uncertainty=std_oSNR/np.sqrt(N)

    thresholds_array=np.array(thresholds)

    mask=thresholds_array>0

    plt.semilogx(
        thresholds_array[mask],
        mean_oSNR[mask],
        label='iSNR = ' + str(isnr) + ' dB'
    )
    
    plt.fill_between(
        thresholds_array[mask],
        mean_oSNR[mask] - uncertainty[mask],
        mean_oSNR[mask] + uncertainty[mask],
        alpha=0.2
    )
    
    # Mean SURE threshold over all extracts
    sure_thresholds_array = np.array(all_sure_thresholds[isnr])
    
    mean_sure_threshold = np.mean(sure_thresholds_array)
    
    if N > 1:
        std_sure_threshold = np.std(sure_thresholds_array, ddof=1)
    else:
        std_sure_threshold = 0

    sure_uncertainty = std_sure_threshold / np.sqrt(N)
    
    # Plot mean SURE threshold as a red vertical line
    plt.axvline(
        mean_sure_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Mean SURE threshold, iSNR = ' + str(isnr) + ' dB'
    )
    
    # Plot uncertainty on mean SURE threshold as a red vertical band
    sure_min = mean_sure_threshold - sure_uncertainty
    sure_max = mean_sure_threshold + sure_uncertainty
    
    # Avoid invalid values on semilogx
    sure_min = max(sure_min, np.min(thresholds_array[mask]))
    
    plt.axvspan(
        sure_min,
        sure_max,
        color='red',
        alpha=0.15
    )

    for i in range(len(thresholds_array)):
        CSV_export.append([
            isnr,
            thresholds_array[i],
            mean_oSNR[i],
            std_oSNR[i],
            uncertainty[i],
            mean_sure_threshold,
            std_sure_threshold,
            sure_uncertainty
        ])

plt.xlabel('Threshold')
plt.ylabel('Mean oSNR')
plt.title('Mean oSNR - Soustraction Spectrale')
plt.grid(True,which='both')
plt.legend()
plt.savefig(path+"/mean_oSNR_soustraction_spectrale.jpg", dpi=300, bbox_inches="tight")
plt.show()


#Export du CSV du graphe
CSV_export=[]

CSV_export.append([
    'iSNR',
    'Threshold',
    'Mean oSNR',
    'Standard deviation oSNR',
    'Uncertainty oSNR',
    'Mean SURE threshold',
    'Std SURE threshold',
    'Uncertainty SURE threshold',
    'Number of extracts'
])

for isnr in params['isnr']:

    oSNR_array=np.array(all_oSNR[isnr])

    mean_oSNR=np.mean(oSNR_array,axis=0)

    if N>1:
        std_oSNR=np.std(oSNR_array,axis=0,ddof=1)
    else:
        std_oSNR=np.zeros(len(mean_oSNR))

    uncertainty=std_oSNR/np.sqrt(N)

    thresholds_array=np.array(thresholds)

    sure_thresholds_array = np.array(all_sure_thresholds[isnr])
    
    mean_sure_threshold = np.mean(sure_thresholds_array)
    
    if N > 1:
        std_sure_threshold = np.std(sure_thresholds_array, ddof=1)
    else:
        std_sure_threshold = 0

    sure_uncertainty = std_sure_threshold / np.sqrt(N)

    for i in range(len(thresholds_array)):
        CSV_export.append([
            isnr,
            thresholds_array[i],
            mean_oSNR[i],
            std_oSNR[i],
            uncertainty[i],
            mean_sure_threshold,
            std_sure_threshold,
            sure_uncertainty,
            N
        ])

with open(path+'/Mean oSNR graph data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(CSV_export)

with open(path+'/Mean oSNR Soustraction Spectrale.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(CSV_export)

'''CSV Export format:
Column 1 --> iSNR values
Column 2 --> Thresholds
Column 3 --> Mean oSNR over all extracts
Column 4 --> Standard deviation over all extracts
Column 5 --> Uncertainty on the mean oSNR = standard deviation / sqrt(number of extracts)
Column 6 --> Mean SURE threshold
Column 7 --> Standard deviation of SURE threshold
Column 8 --> Uncertainty on mean SURE threshold = standard deviation / sqrt(number of extracts)
'''