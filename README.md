[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Auteurs : Moussalli Soukaina, Fikri Ahmed, Bakri Mohammed
# Project Documentation

##  Fonctionnalités

###  **Phase 1 - Érosion simple**
- Test systématique de différents noyaux d'érosion (3×3, 5×5, 7×7)
- Test de différentes itérations (1, 2, 3)
- Génération automatique de comparaisons

###  **Phase 2 - Réduction sélective avancée**
- Détection automatique des étoiles avec `DAOStarFinder`
- Création de masques binaires avec lissage gaussien
- Réduction localisée uniquement sur les étoiles
- Préservation intégrale du fond de nébuleuse

###  **Phase 3 - Extensions (dépassement des objectifs)**
- **Interface graphique unifiée** (`app.py`) avec visualisation temps réel
- **Comparateur interactif** avant/après avec curseur de partage
- **Réduction multi-taille** adaptative (petites/moyennes/grandes étoiles)
- **Traitement par lots** pour automatisation
- *Carte de différence** colorée pour analyse

### Commandes d'execution :

- python erosion.py examples/HorseHead.fits results/
- python erosion.py examples/M31.fits results/

- python phase2_upgraded.py examples/HorseHead.fits results_phase2/
- python phase2_upgraded.py examples/M31.fits results_M31/


## Installation


### Virtual Environment

It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### Dependencies
```bash
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install [package-name]
```

## Usage


### Command Line
```bash
python main.py [arguments]
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Examples files
Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.
- Example 1 : `examples/HorseHead.fits` (Black and whiteFITS image file for testing)
- Example 2 : `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3 : `examples/test_M31_raw.fits` (Color FITS image file for testing)






