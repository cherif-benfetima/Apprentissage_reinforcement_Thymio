# Exo 4 - L experimentation

## 1. Objectif

L exercice 4 demande de valider experimentalement deux objectifs :

1. apprendre au robot a eviter les obstacles,
2. modifier le programme pour apprendre a avancer en absence d obstacle.

Le rendu attendu contient :

- une copie du script,
- une video de demonstration,
- les poids finaux du modele.

## 2. Script utilise

Le script principal est :

- exo4_experimentation.py

Il reprend l architecture des exercices precedents et ajoute un mode experimentation avec trois taches :

- avoid : apprentissage de l evitement,
- forward : apprentissage de l avance sans obstacle,
- both : apprentissage combine.

## 3. Point cle de correction du modele

Pour rendre possible la consigne "avancer en absence d obstacle", une entree de biais a ete ajoutee.

Modele initial (limite) :

- y = W x, avec W en (2 x 5)

Probleme : si x = [0,0,0,0,0], alors y = 0, donc impossible d apprendre une action par defaut.

Modele corrige :

- x_prime = [100, x1, x2, x3, x4, x5]
- y = W x_prime, avec W en (2 x 6)

Cette correction permet d apprendre "avancer" meme quand les capteurs detectent peu ou pas d obstacle.

## 4. Regle d apprentissage

La mise a jour suit la regle de Hebb :

- w_ij <- w_ij + alpha * y_i * x_j

Implementation pratique :

- normalisation numerique : x_norm = x/100 et y_norm = y/100,
- saturation des poids dans [-1.5, 1.5].

## 5. Enseignement et execution

Le script supporte deux modes d enseignement :

1. enseignement manuel (boutons du Thymio)
- avant -> [100, 100]
- arriere -> [-100, -100]
- gauche -> [-100, 100]
- droite -> [100, -100]

2. enseignement automatique (option --auto-teach)
- genere une action enseignee selon la tache active.

Important :

- pour l apprentissage "forward", le code n utilise pas last_informative_x,
- cela evite d associer "avance" a un ancien etat ou il y avait un obstacle.

## 6. Commandes utilisees

### 6.1 Simulation

1. Exo 4 complet (eviter + avancer)

python exo4_experimentation.py --task both --auto-teach --cycles 40 --save-prefix exo4_both

2. Seulement evitement

python exo4_experimentation.py --task avoid --auto-teach --cycles 40 --save-prefix exo4_avoid

3. Seulement avance sans obstacle

python exo4_experimentation.py --task forward --auto-teach --cycles 40 --save-prefix exo4_forward

### 6.2 Robot reel

1. Apprentissage manuel (sans auto-teach)

python exo4_experimentation.py --real --task both --alpha 0.03 --save-prefix exo4_both_real

2. Apprentissage automatique

python exo4_experimentation.py --real --task both --alpha 0.03 --auto-teach --save-prefix exo4_both_real

Arret : bouton central du robot.

## 7. Fichiers de poids produits

A la fin de chaque execution, le script enregistre :

- un fichier NPY : poids du modele,
- un fichier TXT : poids lisibles pour le rapport.

Exemple :

- exo4_poids_finaux.npy
- exo4_poids_finaux.txt

## 8. Validation des deux consignes

### 8.1 Consigne 1 : eviter les obstacles

Validation si en mode AUTO :

- obstacle a gauche -> rotation a droite,
- obstacle a droite -> rotation a gauche,
- obstacle devant -> freinage/recul.

### 8.2 Consigne 2 : avancer sans obstacle

Validation si en zone sans obstacle :

- le robot apprend une avance par defaut,
- grace a l entree de biais (W en 2x6).

## 9. Conclusion

L exercice 4 est valide :

1. le robot apprend a eviter les obstacles,
2. le programme a ete adapte pour apprendre a avancer en absence d obstacle,
3. les poids finaux sont sauvegardes pour le rendu,
4. le script fonctionne en simulation et sur robot reel.