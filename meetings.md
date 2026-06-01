# Compte-rendus des meetings
## Meeting du 01/06/2026 (Modifiée 01/06/2026)

**Fait :**
- Experiences DEMUCS et SGMSE+ pour des extrait vocaux et musicaux
- Comparaison des SNR de sortie entre soustraction spectrale, DEMUCS, et SGMSE+
- Création de Overleaf et de la bibliographie

**À faire :**
- Créer un script qui prend les valeurs de SNR pour toutes les méthodes et qui renvoi le graphe comparatif
- Mettre en place des tests d'utilisation de SURE
- Accéder aux serveurs de calculs du laboratoire
- Moyenner chaque courbe SNR et calculer la déviation standard en utilisant 10 extraits musicaux
- Préciser les fréquences d'échantillonage et les paramètres stft

**Questions :**
- Est ce que je cré un graphe pour la voix et un pour les extraits musicaux
- Pourquoi DEMUCS produit une oSNR beaucoups moins bonne que la soustraction spectrale pour les extraits vocaux alors qu'elle a un son plus propre

## Meeting du 25/05/2026 (Modifiée 22/05/2026)

**Fait :**
- Création du GitHub et de la structure du dépot
- Création de l'environnement Python et installation des outils de base
- Reprise du TP de soustraction spectrale et applications dans un notebook
- Création des scripts et paquets Python
- Mise en place de l'experience oracle
- Lecture sur les débruiteurs Deep

**À faire :**
- Création du fichier Overleaf avec bibliographie
- Remettre en place le debruitage par soustraction spectrale a seuil constant et comparer avec le débruitage gain Wiener
- Modification de environment.yml pour installer deepinv sans changer de version Pytorch
- Installation et observation de DEMUCS et SGMSE+
- Voir s'il est possible d'utiliser les GPU du laboratoire pour les débruiteurs Deep
- Installer tous les systèmes sur le PC pour pouvoir utiliser un GPU personnel au besoin

**Questions :**
- Quels types de musique utiliser préférentiellement, musique instrumentale, musique avec vocaux, ou voix seule ?
- Validation de la première expérience : iSNR --> ajout de bruit blanc $\sigma$ au son pur --> débruitage a gain de type Wiener pour plusieurs seuils --> graphe de oSNR en fonction du seuil