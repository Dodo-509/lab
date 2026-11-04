# TP Computer Vision : Benchmark d'Architectures de Réseaux de Neurones sur MNIST

## 1. Vue d'Ensemble du Projet & Architectures

Ce projet de laboratoire évalue et compare différentes architectures de Deep Learning et Vision par Ordinateur pour la classification de chiffres manuscrits sur le jeu de données **MNIST**. Les modèles comparés comprennent :
- **CNN Custom :** Un réseau convolutif léger entraîné à partir de zéro (*from scratch*).
- **Faster R-CNN (Backbone ResNet-50) :** Un framework de détection d'objets adapté à la classification d'objet unique.
- **AlexNet (Fine-Tuned) :** Un réseau convolutif classique pré-entraîné sur ImageNet.
- **VGG16 (Fine-Tuned) :** Un réseau convolutif profond pré-entraîné sur ImageNet.
- **Vision Transformer (ViT from Scratch) :** Une architecture basée sur le mécanisme d'attention, adaptée à la classification d'images via le découpage en patchs (*patch embeddings*) et l'auto-attention multi-têtes (MHSA).

---

## 2. Tableau Comparatif des Résultats

Toutes les expériences ont été évaluées avec PyTorch et `torchmetrics` dans un environnement GPU sur Google Colab (NVIDIA T4) selon un protocole expérimental réproductible.

| Architecture | Accuracy (Test) | F1-Score (Test) | Loss (Test) | Temps d'Entraînement (s) | Temps d'Évaluation (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Custom** | 0.9861 | 0.9860 | 0.0401 | ~120.00 | **1.51** |
| **Faster R-CNN** | 0.9780 | 0.9775 | 0.0668 | 2744.39 | ~18.50 |
| **AlexNet (Fine-Tuned)** | 0.9925 | 0.9925 | 0.0257 | 552.70 | 26.13 |
| **VGG16 (Fine-Tuned)** | **0.9950** | **0.9950** | **0.0173** | 3008.75 | 68.40 |
| **ViT (from scratch)** | 0.8792 | 0.8773 | 0.3870 | 1632.25 | 28.90 |

---

## 3. Analyse Comparative & Synthèse

1. **Efficacité du Transfer Learning (VGG16 & AlexNet) :**
   - **VGG16** a obtenu l'accuracy la plus élevée (**99.50%**) et la perte la plus faible (**0.0173**), prouvant que les cartes de caractéristiques pré-entraînées sur de grands jeux de données (ImageNet) se transfèrent très bien à des tâches visuelles plus simples.
   - **AlexNet** a offert le meilleur compromis entre vitesse et performance : il atteint **99.25%** d'accuracy pour un temps d'entraînement de **552.70 secondes** (plus de 5 fois plus rapide que VGG16).

2. **CNN Custom vs. Modèles Pré-entraînés :**
   - Le **CNN Custom** a obtenu d'excellentes performances (**98.61% d'Accuracy**) tout en étant extrêmement rapide à évaluer (**1.51 secondes**) et à entraîner. Pour des jeux de données de basse résolution comme MNIST ($28 \times 28$), les architectures légères sur mesure restent particulièrement adaptées au déploiement embarqué.

3. **Contraintes de Faster R-CNN pour la Classification Pure :**
   - **Faster R-CNN** s'est avéré très lourd sur le plan calculatoire pour une simple tâche de classification (**2744.39s** d'entraînement). Les chiffres MNIST ne nécessitant pas de réseau complexe de proposition de régions (RPN), l'utilisation d'un modèle de détection d'objets introduit un surcoût inutile.

4. **Vision Transformer (ViT from Scratch) :**
   - Le **ViT entraîné from scratch** a atteint **87.92% d'Accuracy**. Contrairement aux CNNs, les Transformers ne possèdent pas de biais inductifs spatiaux (tels que la translation équivariante et la localité). Par conséquent, entraîner un Transformer à partir de zéro sur un petit dataset sans pré-entraînement mène à des performances inférieures à celles d'un réseau convolutif.

---

## 4. Ce que j'ai appris 
> Au cours de ce TP, j'ai développé des compétences pratiques et théoriques sur la conception, l'entraînement et le benchmarking de modèles de Deep Learning avec PyTorch :
>
> 1. **Maîtrise des Frameworks de Deep Learning :** J'ai mis en place des pipelines systématiques sous PyTorch, `torchvision` et `torchmetrics` pour évaluer rigoureusement les modèles selon les métriques standards (Accuracy, F1-Score Macro, Loss, Temps d'exécution).
> 2. **Compréhension des Compromis Architecturaux :** J'ai appris que le choix d'un modèle dépend fortement des contraintes du problème. Alors que les architectures profondes comme VGG16 offrent une précision maximale, les CNNs légers garantissent une efficacité imbattable pour des tâches simples.
> 3. **Adaptation de la Détection d'Objets :** L'adaptation de Faster R-CNN a mis en évidence l'importance du redimensionnement d'image ($128 \times 128$), de la gestion des canaux (RGB) et du sous-échantillonnage lors de contraintes matérielles ou de temps sur Colab.
> 4. **Fondements des Vision Transformers :** L'implémentation d'un ViT from scratch m'a permis de comprendre concrètement le découpage en patchs, les positional embeddings et l'auto-attention multi-têtes (MHSA). Cela a démontré pourquoi les Transformers nécessitent un pré-entraînement à grande échelle ou une forte régularisation pour égaler les réseaux convolutifs.
> 5. **Reproductibilité Scientifique :** La fixation des graines aléatoires (*random seeds*) et la structuration des tests m'ont sensibilisé à l'importance de la réproductibilité en apprentissage automatique.