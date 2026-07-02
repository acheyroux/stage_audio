# Experience Seuillage par SURE
### Contexte
Mesure du SNR de sortie par débruitage SGMSE+

### Instructions d'utilisation
1. Dans Miniforge Prompt taper la commande ```gdown 16K4DUdpmLhDNC7pJhBBc08pkSIn_yMPi``` pour installer le checkpoint SGMSE+
2. Placer le fichier ```.ckpt``` dans le dossier ```/Data/```
3. Ouvrir le fichier ```params.txt``` pour confirmer les paramètres ou les modifier dans le cas échéant
4. Lancer le script a l'aide de la commande ```python experience_sgmse.py``` dans l'environnement (audio-denoising) de Miniforge Prompt
5. Recueillir les données

### Résultats
Nous remarquons comme pour le débruiteur DEMUCS que pour un nombre d'étapes important ```N=30```, le debruiteur SGMSE+ filtre le bruit blanc pour l'extrait sonore vocal mais ne filtre pas correctement le fichier musical car c'est un modèle basé sur la parole.