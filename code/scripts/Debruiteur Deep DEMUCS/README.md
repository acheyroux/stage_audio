# Expérience : Débruitage avec DEMUCS
### Decription
Ce script ajoute du bruit blanc gaussien artificiel sur un fichier sonore pur et le débruite a l'aide du débruiteur DEMUCS.
Il génère des fichiers ```noisy_sound.wav``` et ```denoised_sound.wav``` pour chaque SNR d'entrée, une copie des paramètres utilisés, et un fichier text avec les SNR après débruitage correspondants aux SNR d'entrée.

### Instructions d'utilisation
1. Placer un fichier sonore de format ```___.wav``` dans le dossier ```/stage_audio/data/``` (le créer si nécessaire)
2. Ouvrir le fichier ```params.txt``` pour confirmer les paramètres ou les modifier dans le cas échéant
3. Lancer le script a l'aide de la commande ```python experience_demucs.py``` dans l'environnement (audio-denoising) de Miniforge Prompt
4. Recueillir les données

### Résultats
Nous remarquons que le debruiteur DEMUCS filtre le son vocal correctement mais produit un résultat incohérent pour le son musical, possiblement car il est entrainé sur de la parole et non de la musique. En effet, pour un son purement instrumental, 