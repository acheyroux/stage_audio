# Expérience : Comparaison des graphes oSNR
### Decription
Ce script cré les graphes de comparaison entre les méthodes de débruitage par soustraction spectrale ainsi que les méthodes de débruitage deep (DEMUCS et SGMSE+) par l'intermédiaire d'auto-paramétrisation avec SURE (librairie deepinv). Il trace aussi l'estimation du niveau de bruit blanc à l'aide de la minimisation UNSURE et le vrai niveau de bruit.

### Remarques
Cette expérience nécessite un GPU dédié (non-intégré) et peu atteindre la limite Vram si les paramètres ne sont pas modifiés.

### Instructions d'utilisation
1. Placer le dataset MUSDB18-HQ ou autre base de donnée de format similaire dans ```/data/```
2. Ouvrir le fichier ```params.txt``` pour confirmer les paramètres ou les modifier dans le cas échéant
3. Lancer le script a l'aide de la commande ```python experience_osnr.py``` dans l'environnement (audio-denoising) de Miniforge Prompt
4. Recueillir les données

### Paramètres
- track_samples : nombre d'extraits à traiter
- sound_type : nom du fichier ```.wav``` standardisé (exemple mixture.wav ou vocals.wav pour MUSDB18-HQ)
- sgmse_duration : durée de traitement du signal par SGMSE+ dans le cas de limite Vram (10s max)
- sgmse_sure_N : sampler settings pour SGMSE+ (cf documentation)
- sgmse_sure_lr : taux d'apprentissage de SURE pour SGMSE+
- sgmse_sure_steps : étapes d'apprentissage de SURE pour SGMSE+
- demucs_sure_steps : idem pour DEMUCS
- demucs_sure_batch_size : nombre de traitements simultanés de SURE pour DEMUCS
- demucs_sure_lr : idem pour DEMUCS
- unsure_steps : étapes d'apprentissage pour UNSURE
- unsure_batch_size : nombre de traitements simultanés de UNSURE
- unsure_lr : taux d'apprentissage de UNSURE
