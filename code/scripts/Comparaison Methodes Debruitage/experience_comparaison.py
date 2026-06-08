import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#Creation du dossier de l'experience
path="../../../results/"+datetime.now().strftime("%Y%m%d_%H%M")+"_experience_comparaison"
os.mkdir(path)
ckpt_path="../../../Data/train_wsj0_2cta4cov_epoch=159.ckpt"
os.mkdir(path+"/SNR Data")

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

#Prise en compte des donnnees et copiage des donnees initiales
for file in os.listdir("SNR Data"):
    shutil.copyfile(os.path.join("SNR Data", file),os.path.join(path+"/SNR Data", file))


present_iSNRs=[]
#Format de chaque liste data : [[iSNR_1,oSNR_1,sigma_1],...]
#DEMUCS
DEMUCS_data={}
with open("SNR Data/SNR DEMUCS.txt",'r') as file:
    for full_line in file:
        line=full_line.strip()
        val=line.strip().split('\t')
        item_isnr=float(val[1].split('=')[1])
        DEMUCS_data[item_isnr]=[item_isnr,float(val[2].split('=')[1]),float(val[0].split('=')[1])]
        if not(item_isnr in present_iSNRs):
            present_iSNRs.append(item_isnr)
#SGMSE+
SGMSE_data={}
with open("SNR Data/SNR SGMSE+.txt",'r') as file:
    for full_line in file:
        line=full_line.strip()
        val=line.strip().split('\t')
        item_isnr=float(val[1].split('=')[1])
        SGMSE_data[item_isnr]=[item_isnr,float(val[2].split('=')[1]),float(val[0].split('=')[1])]
        if not(item_isnr in present_iSNRs):
            present_iSNRs.append(item_isnr)
#Soustraction Spectrale
Specsub_data={}
with open("SNR Data/SNR Soustraction Spectrale.csv", "r") as csvfile:
    data = csv.reader(csvfile)
    for data_list in data:
        if len(data_list)!=0:
            if data_list[0]=='Threshold':
                thresholds=[float(i) for i in data_list[1:]]
            else:
                item_isnr=float(data_list[0])
                if not(item_isnr in present_iSNRs):
                    present_iSNRs.append(item_isnr)
                Specsub_data[item_isnr]=[item_isnr,[float(j) for j in data_list[1:]]]

#SURE Sosutraction Spectrale
SURE_thresh=-1
with open("SNR Data/Seuil SURE.txt",'r') as file:
    for full_line in file:
        line=full_line.strip()
        if line:
            val=line.strip().split('\t')
            SURE_thresh=float(val[1].strip())

for isnr in present_iSNRs:
    fig,ax=plt.subplots()
    if isnr in Specsub_data:
        ax.plot(thresholds,Specsub_data[isnr][1],label='Spectral Sub')
        index_oracle=np.argmax(Specsub_data[isnr][1])
        oracle_thres,oracle_oSNR=thresholds[index_oracle],Specsub_data[isnr][1][index_oracle]
        ax.scatter(oracle_thres,oracle_oSNR,c='r',)
        ax.annotate('Oracle', (oracle_thres,oracle_oSNR),xytext=(oracle_thres+.2,oracle_oSNR+.2),arrowprops=dict(arrowstyle="->"))
        if SURE_thresh!=-1:
            SURE_index=np.argmin(np.abs(np.array(thresholds) - SURE_thresh))
            SURE_oSNR=Specsub_data[isnr][1][SURE_index]
            ax.scatter(SURE_thresh,SURE_oSNR,c='r',)
            ax.annotate('SURE', (SURE_thresh,SURE_oSNR),xytext=(SURE_thresh+.2,SURE_oSNR+.2),arrowprops=dict(arrowstyle="->"))
    if isnr in DEMUCS_data:
        ax.plot(thresholds,np.ones_like(thresholds)*DEMUCS_data[isnr][1],label='DEMUCS')
    if isnr in SGMSE_data:
        ax.plot(thresholds,np.ones_like(thresholds)*SGMSE_data[isnr][1],label='SGMSE+')
    ax.set_title("Output SNR for Various Denoising Methods")
    ax.set_xlabel(r"Threshold $\lambda$")
    ax.set_ylabel("oSNR (dB)")
    ax.grid(True)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(path+"/isnr_"+str(isnr)+".jpg", dpi=300, bbox_inches="tight")

with open("params.txt", 'r') as file:
    content = file.read()
with open(path+"/params.txt", 'w') as otherfile:
    otherfile.write(content)

