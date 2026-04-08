# Exo 3 - La regle de Hebb

## 1. Objectif

L objectif de cet exercice est de mettre a jour les poids du reseau de neurones en appliquant explicitement la regle de Hebb.

Le robot conserve la meme architecture que dans les exercices precedents (5 entrees capteurs, 2 sorties moteurs), mais la phase d apprentissage suit maintenant l algorithme 3.

## 2. Modele du reseau

Le reseau utilise :

- vecteur d entree : x = [x1, x2, x3, x4, x5]^T
- vecteur de sortie : y = [y1, y2]^T
- equation de sortie : y = W x

Avec :

- y1 : vitesse roue gauche
- y2 : vitesse roue droite
- sorties bornees dans [-100, 100]
- capteurs normalises dans [0, 100]

## 3. Algorithme 3 applique

L algorithme 3 indique :

1. lire x issu des capteurs
2. pour j dans {1,2,3,4,5} :
- w1j <- w1j + alpha * y1 * xj
- w2j <- w2j + alpha * y2 * xj

Dans le code, cette mise a jour est implementee dans `hebb_update(...)` avec une boucle explicite sur les 5 capteurs.

## 4. Commande enseignee (utilisation des boutons)

Pour fournir une action de reference pendant l apprentissage, les boutons directionnels sont associes a :

1. bouton avancer : [100, 100]
2. bouton reculer : [-100, -100]
3. bouton gauche : [-100, 100]
4. bouton droite : [100, -100]

La commande enseignee est appliquee aux moteurs et sert de signal y dans la mise a jour Hebb.

## 5. Details d implementation

Le script utilise :

- normalisation numerique pour eviter une divergence rapide :
- x_norm = x / 100
- y_norm = y / 100

- saturation des poids dans [-1.5, 1.5]
- memorisation du dernier x informatif pour eviter les mises a jour nulles quand x = [0,0,0,0,0]

## 6. Fichier et execution

Fichier Python de l exercice :

- exo3_regle_hebb.py

Commandes principales :

1. simulation

python exo3_regle_hebb.py --cycles 20

2. robot reel

python exo3_regle_hebb.py --real

3. ajuster le taux d apprentissage

python exo3_regle_hebb.py --real --alpha 0.03

4. initialisation aleatoire (optionnelle)

python exo3_regle_hebb.py --random-init

## 7. Lecture des sorties console

A chaque cycle, le script affiche :

- mode = AUTO, HEBB, ou HEBB_SKIP
- x : valeurs capteurs
- y_net : sortie du reseau avant commande enseignee
- y_apply : commande appliquee aux moteurs

Interpretation :

- AUTO : pas d apprentissage, le robot suit y = W x
- HEBB : apprentissage effectif applique
- HEBB_SKIP : apprentissage ignore faute de situation capteur exploitable

## 8. Validation pratique

Pour valider l exercice 3 :

1. realiser plusieurs episodes d enseignement en mode HEBB
2. arreter l enseignement (AUTO)
3. verifier que le comportement appris est coherent avec :
- obstacle gauche -> tourner a droite
- obstacle droite -> tourner a gauche
- obstacle devant -> reculer ou freiner

## 9. Conclusion

L exercice 3 est valide si la matrice W evolue au cours des interactions et produit un comportement autonome coherent en mode AUTO, conformement a la regle de Hebb.