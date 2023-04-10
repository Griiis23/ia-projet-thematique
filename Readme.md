# Serveur de traitement d'image

## Installation

Il suffit de récupérer le projet
```
git clone https://github.com/Griiis23/ia-projet-thematique.git
```

Installer les requirements python
```
pip install -r requirements.txt
```

En cas de problème avec opencv, réinstaller opencv
```
pip uninstall opencv-python-headless opencv-python
pip install opencv-python==4.7.0.72
```

Modifier le fichier de configuration
```
nano .env
```

Modifier le fichier des dossards
```
nano bibs.txt
```

Lancer le serveur
```
python3 server.py
```
