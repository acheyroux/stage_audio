# Stage L3 : Comparaison de méthodes de débruitage audio musical

# Français
## Contexte

Cette page GitHub contient l'ensemble des codes et ressources pour le stage de Comparaison de méthodes de débruitage audio musical.

## Description des fichiers

- environment.yml : description des paquets Python de l'environnement nécessaires pour le lancement des scripts (installation mamba)
- meetings.md : compte rendu et résumé des entretiens hebdomadaires
- .gitignore : description des fichiers non maintenus sur cette page GitHub
- results/ : résultats des expériences et scripts
- code/ : ensemble des codes et scripts
- data/ : données et ressources initiales
- biblio/ : références et bibliographie

## Utilisation
#### Création de l'environnement
1. Télécharger Miniforge : https://github.com/conda-forge/miniforge
2. Ouvrir un terminal et suivre les instructions pour dupliquer le dépot : https://docs.github.com/fr/repositories/creating-and-managing-repositories/cloning-a-repository
3. Ouvrir un terminal Miniforge Prompt et installer Jupyter Lab ```mamba install -c conda-forge jupyterlab```
4. Dans le terminal Miniforge Prompt naviguer au dossier /stage_audio
5. Taper consécutivement les commandes ```mamba env create -f environment.yml``` puis ```mamba activate audio-denoising```

#### Exécution du code
- Pour visualiser les notebooks, ouvrir Jupyter Lab avec la commande ```jupyter lab``` dans le dossier stage_audio du terminal Miniforge Prompt
- Pour lancer les experiences et scripts Python, executer les fichiers ```___.py```

# English
## Context

This GitHub page contains all the code and ressources for the internship - Comparison of musical audio-denoising methods.

## Object descriptions

- environment.yml : description of required Python libraries in the environment (mamba installation)
- meetings.md : weekly meeting reports
- .gitignore : description of objects not tracked by Git
- results/ : script experience results
- code/ : Python scipts and code
- data/ : ressources and data
- biblio/ : citation references

## How to use
#### Environment setup
1. Install Miniforge : https://github.com/conda-forge/miniforge
2. Open a terminal and follow the instructions to duplicate the repository : https://docs.github.com/fr/repositories/creating-and-managing-repositories/cloning-a-repository
3. Open a Miniforge Prompt terminal and install Jupyter Lab ```mamba install -c conda-forge jupyterlab```
4. In the Miniforge Prompt navigate to the /stage_audio folder
5. Input these commands: ```mamba env create -f environment.yml``` followed by ```mamba activate audio-denoising```
6. Launch Jupyter Lab with this command ```jupyter lab```

#### How to run code
- For notebooks, launch Jupyter Lab from the Miniforge Prompt using ```jupyter lab``` while in the stage_audio folder
- For scripts and experiments, launch the desired Python scripts ```___.py```