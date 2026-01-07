Phase 3 – Prolongements

Deux prolongements ont été réalisés afin d’améliorer l’utilisation de l’algorithme de réduction d’étoiles.

Interface Utilisateur
Une interface graphique simple permet de charger un fichier FITS et d’ajuster les paramètres de réduction (taille du noyau et nombre d’itérations) sans modifier le code.
Exécution;
python3 gui_star_reduction.py

Utilisation :

Charger un fichier FITS (ex : examples/HorseHead.fits)

Ajuster les paramètres

Appliquer la réduction

L’image finale est automatiquement sauvegardée dans le dossier results/.

Batch Processing

Un mode de traitement par lot permet d’appliquer automatiquement la réduction d’étoiles à un ensemble d’images FITS.

Exécution :

python3 batch_star_reduction.py inputs outputs


inputs/ : dossier contenant les fichiers .fits

outputs/ : dossier où sont enregistrés les résultats

Ce mode permet de traiter plusieurs images avec les mêmes paramètres.