# Exo 2 - La recompense

## 1. Objectif

Introduire la notion d apprentissage dans le modele en modifiant la matrice des poids W a partir d un retour utilisateur par boutons.

Le principe est :
- le robot calcule d abord y = W x,
- l utilisateur peut corriger le comportement en appuyant sur un bouton,
- la correction est utilisee pour mettre a jour W.

## 2. Rappel du modele

Le reseau conserve la structure de l exercice 1 :

- entree : x = [x1, x2, x3, x4, x5]^T
- sortie : y = [y1, y2]^T
- equation : y = W x

Avec :
- y1 : vitesse roue gauche
- y2 : vitesse roue droite
- sorties saturees dans [-100, 100]
- capteurs normalises dans [0, 100]

## 3. Algorithme 2 (recompense par boutons)

Le comportement enseigne est associe aux 4 boutons :

1. bouton avancer : y1 = 100, y2 = 100
2. bouton reculer : y1 = -100, y2 = -100
3. bouton tourner a gauche : y1 = -100, y2 = 100
4. bouton tourner a droite : y1 = 100, y2 = -100

Priorite appliquee si plusieurs boutons sont appuyes :
- avancer > reculer > gauche > droite

## 4. Loi d apprentissage utilisee

Quand un bouton est appuye, la commande y_reward est utilisee pour l apprentissage :

Delta W = alpha * (y_reward * x^T)

Dans le code :
- x et y_reward sont d abord normalises (division par 100),
- puis W est mis a jour,
- enfin W est borne dans [MIN_WEIGHT, MAX_WEIGHT] = [-1.5, 1.5].

## 5. Robustesse ajoutee dans l implementation

Pour eviter un apprentissage nul quand x = [0,0,0,0,0] :

1. on detecte si la situation capteur est informative (seuil minimal),
2. si besoin, on reutilise le dernier x informatif memorise pendant une courte duree,
3. sinon on passe en mode LEARN_SKIP.

## 6. Modes d execution

Fichier Python utilise :
- exo2_recompense.py

Commandes :

1. simulation

python exo2_recompense.py --cycles 20

2. robot reel

python exo2_recompense.py --real

3. regler alpha

python exo2_recompense.py --real --alpha 0.03

4. initialisation aleatoire (optionnelle)

python exo2_recompense.py --random-init

## 7. Lecture des logs

A chaque cycle, le script affiche :
- mode : AUTO, LEARN, ou LEARN_SKIP
- x : capteurs normalises
- y_net : sortie du reseau (avant correction)
- y_apply : commande appliquee aux moteurs

Interpretation :
- AUTO : le robot agit avec son apprentissage courant,
- LEARN : un bouton enseigne une action et met a jour W,
- LEARN_SKIP : bouton appuye mais pas de situation capteur exploitable.

## 8. Validation finale conseillee

Apres apprentissage, faire un test sans appuyer sur les boutons (AUTO uniquement) et verifier :

1. obstacle a gauche -> rotation a droite
2. obstacle a droite -> rotation a gauche
3. obstacle devant -> recul/freinage

## 9. Conclusion

L exercice 2 est valide :
- la recompense par boutons est implementee,
- les poids W evoluent en ligne,
- le robot peut apprendre un comportement d evitement a partir du retour utilisateur.
