## Plan d'attaque - Machine Learning, projet 2 - Road Segmentation

## Problème
Identifier les routes d'une image satellite.

## Méthode
Fully Convolutional Network (FCN) (apparemment c'est state-of-the-art: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

## Principe de la méthode
Lier l'image d'input à une image de même taille dont chaque pixel est la prédiction du modèle pour le pixel d'input correspondant (dans notre cas: to road or not to road).

## Objectif
Obtenir une submission telle que chaque batch de 16 pixels par 16 pixels soit assigné.

## Roadblocks

* Roadblock 1: Les images du training set sont de taille 400x400 pixels, les images du test set sont de taille 608x608. Si on décide d'augmenter notre training set (et qu'on trouve des images supplémentaires...), ces images auront une taille différente. Le FCN peut seulement accepter des input de taille fixe.

	SOLUTION: Entraînons le modèle sur des images de taille 400 * 400, puis sur les images du test set/training set additionnel, séparons les images en patchs de 400 * 400 pixels, en gardant la position du pixel en haut à gauche en mémoire. On pourra donc reconstruire les images entières en superposant ces patchs, prenant la moyenne des pixels d'output superposés.

* Roadblock 2: Les outputs doivent être des prédictions pour chaque batch de 16x16 pixels.

	SOLUTION 1: Faire un FCN qui lie une prédiction pour chaque pixel, prendre la moyenne de chaque batch de 16x16 pixels.

	SOLUTION 2: Modifier le FCN pour qu'il lie une prédiction pour chaque batch de 16x16 pixels. Cela impliquerait de d'abord downsample les images groundtruth du training set (prendre la moyenne de chaque batch de 16x16 pixels) et d'entraîner le modèle sur ça, et ça impliquera aussi de restreindre les possibilités de découpage des images du test set/training set additionnel, forçant le découpage à 16xk pixels (avec k un entier).

	- `[Manuel]` Je pense que la solution 1 est mieux si on peut trouver assez de données pour que la prédiction soit assez fiable. Sinon, la solution 2 est mieux selon moi puisque la taille d'output est réduite, possiblement réduisant la taille du training set nécessaire pour avoir une prédiction décente.

## UPDATE - 27.11.2018

- `[Manuel]` Les datasets supplémentaires n'ont pas la même résolution spatiale (~25 pixels/largeur de route pour le dataset donné pour le projet, résolution environ deux fois plus basse pour les datasets supplémentaires). La solution la plus simple selon moi serait de diminuer la résolution du dataset de base (rassembler patches de 2x2 pixels en un pixel) pour avoir un dataset final plus uniforme, cela ne devrait pas changer substantiellement la qualité des prédictions sur le dataset de base, puisqu'on rassemble des patchs de 16x16 pixels pour les prédictions. Tests à effectuer pour vérifier si la qualité des prédiction est meilleure.

- `[Manuel]` Le dataset supplémentaire 1 contient quelques images avec des parties blanches, il serait utile de supprimer les patchs qui sont en majorité blancs.

- `[Manuel]` Le dataset supplémentaire 2 contient beaucoup de routes en terre, à voir s'il est vraiment utile pour le projet...

## UPDATE - 02.12.2018

- `[Manuel]` Premier test de CNN qui semble concluant (fichier preprocess.py), la loss diminue avec les epochs. Hyperparamètres à tester: nombre de filtres, quantité de dropout. Voir si remonter à une image de 200x200 améliore les prédictions.
