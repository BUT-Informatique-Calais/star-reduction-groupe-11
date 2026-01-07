Phase 2 – Réduction sélective des étoiles

Lors de la phase 1, l’érosion globale permettait de réduire la taille des étoiles, mais elle dégradait également le fond de la nébuleuse. Afin de corriger ce problème, une méthode sélective basée sur un masque d’étoiles a été mise en place.

Dans cette phase, un masque binaire des étoiles est d’abord créé à partir de l’image FITS à l’aide d’un seuillage adaptatif. Les pixels correspondant aux étoiles sont représentés en blanc, tandis que le fond reste noir. Afin d’éviter des transitions trop brutales lors du traitement, les contours du masque sont ensuite adoucis grâce à un flou gaussien.

Une version érodée de l’image originale est ensuite calculée. L’image finale est obtenue par interpolation entre l’image originale et l’image érodée, en utilisant le masque d’étoiles selon la formule suivante :

Ifinal = (M × Ierode) + ((1 − M) × Ioriginal)

Cette approche permet de réduire principalement les étoiles tout en préservant davantage les structures diffuses de la nébuleuse, contrairement à l’érosion appliquée sur l’image entière. Les résultats obtenus montrent une amélioration visuelle significative par rapport à la phase 1.
