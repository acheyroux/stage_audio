# Expérience 21/05/2026 : Recherche et debruitage par soustraction spectrale en seuil oracle
### Decription
Ce script ajoute du bruit blanc gaussien artificiel sur un fichier sonore pur et le débruite par soustraction spectrale en seuil oracle.
Il génère un fichier ```noisy_sound.wav```, un fichier ```denoised_sound.wav```, et un graphe montrant le seuil oracle.

Le script multiple effectue le même script sur tous les extraits sonores dans le dossier précisé et cré un fichier ```.csv``` et un graphe moyennant les courbes oSNR trouvés et les valeurs déterminés par SURE

### Instructions d'utilisation - (Rq : Fonctionnement analogue pour le script multiple)
1. Placer un fichier sonore de format ```___.wav``` dans le dossier ```/stage_audio/data/``` (le créer si nécessaire)
2. Ouvrir le fichier ```params.txt``` pour confirmer les paramètres ou les modifier dans le cas échéant
3. Lancer le script a l'aide de la commande ```python experience_oracle.py``` dans l'environnement (audio-denoising) de Miniforge Prompt
4. Recueillir les données

### Résultats
Nous remarquons que le fichier sonore est débruité mais qu'il reste des artéfacts sonores du type clics et bips très nuisants pour l'écoute. Nous concluons donc que la soustraction spectrale en seuil oracle n'est pas nécessairement idéale dans la majorité des cas.