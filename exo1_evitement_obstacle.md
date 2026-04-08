# Exo 1 - Apprentissage par renforcement

## 1. Objectif

L objectif de cet exercice est de valider le modele de reseau de neurones propose pour l evitement d obstacle, puis de l executer de facon periodique sur simulation ou sur robot Thymio.

Le reseau utilise 5 entrees capteurs et 2 sorties moteurs.

## 2. Modele du reseau

### 2.1 Entrees

Le vecteur d entree est :

x = [x1, x2, x3, x4, x5]^T

avec :
- x1 : capteur frontal gauche
- x2 : capteur central
- x3 : capteur frontal droit
- x4 : capteur arriere gauche
- x5 : capteur arriere droit

Les entrees sont bornees dans [0, 100].

### 2.2 Sorties

Le vecteur de sortie est :

y = [y1, y2]^T

avec :
- y1 : vitesse roue gauche
- y2 : vitesse roue droite

Les sorties sont bornees dans [-100, 100].

### 2.3 Equation utilisee

Equation de calcul :

y = W x

avec W une matrice 2 x 5.

## 3. Algorithme applique (periode 100 ms)

A chaque periode T = 0.1 s :

1. Lire les capteurs et construire x.
2. Calculer y = W x.
3. Saturer les sorties dans [-100, 100].
4. Appliquer y1 et y2 aux moteurs gauche et droit.

Ce comportement correspond a l algorithme 1 de l enonce (partie controle periodique).

## 4. Fichier Python de l exercice

Implementation realisee dans :
- exo1_hebb_evitement_obstacle.py

Le script propose deux modes :
- simulation (par defaut)
- robot reel Thymio (option --real)

## 5. Commandes d execution

Depuis le dossier du TP :

- Simulation :
  python exo1_hebb_evitement_obstacle.py

- Robot reel :
  python exo1_hebb_evitement_obstacle.py --real

## 6. Interpretation du comportement

La matrice W est choisie pour obtenir un comportement d evitement :

- obstacle a gauche : rotation vers la droite
- obstacle a droite : rotation vers la gauche
- obstacle frontal : freinage des roues

Le script affiche a chaque cycle :
- les entrees capteurs x
- les commandes moteurs y

Cela permet de verifier la coherence entre perception et action.

## 7. Conclusion

L exercice 1 est valide avec une implementation conforme au modele y = W x et une execution periodique de 100 ms.

La structure est prete pour une extension ulterieure vers un apprentissage Hebbien avec retour binaire (bon/mauvais), si demande dans les exercices suivants.
