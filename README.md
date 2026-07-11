# RCA-GAT-GRU — Détection de service fautif et de type de panne par apprentissage sur graphes temporels

Système de **Root Cause Analysis (RCA)** pour environnements micro-services / Kubernetes, combinant un encodeur de graphes d'attention (**GATv2**) et un module temporel (**GRU bidirectionnel + attention**) pour identifier, à partir de séquences de graphes d'appels inter-services, **quel service est à l'origine d'un incident** et **quel type de panne** est en cours.

Le modèle est multi-tâche : il produit simultanément une prédiction de **service fautif** et une prédiction de **classe de panne**, à partir d'une fenêtre glissante de graphes de communication/ressources/événements.

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture du modèle](#architecture-du-modèle)
- [Pipeline de données](#pipeline-de-données)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Format des données d'entrée](#format-des-données-dentrée)
- [Entraînement](#entraînement)
- [Évaluation et inférence](#évaluation-et-inférence)
- [Artefacts produits](#artefacts-produits)
- [Configuration et hyperparamètres](#configuration-et-hyperparamètres)

- [Licence](#licence)

---

## Aperçu

Dans un système distribué, un incident (latence anormale, taux d'erreur élevé, chute de throughput) se propage souvent d'un service vers ses dépendances. Ce projet modélise chaque **fenêtre temporelle** de l'infrastructure comme un **graphe** :

- les **nœuds** = les services (pods), avec des features de ressources, d'événements et de causalité,
- les **arêtes** = les communications inter-services, avec des features de latence, débit, erreurs et un **score de causalité** calculé heuristiquement,
- une **séquence de graphes consécutifs** (fenêtre glissante) est ensuite encodée dans le temps pour capter la dynamique de propagation de la panne.

Le modèle final répond à deux questions :

1. **Quel service est la cause racine de l'incident ?** (classification multi-classe `y1`)
2. **Quel type de panne est en cours ?** (classification multi-classe `y2`)

---

## Architecture du modèle

Le modèle (`GATGRUMultiTask`, dans [`model.py`](./model.py)) s'articule en quatre blocs :

![Architecture](model_architecture_with_layers.png)

### 1. `GATEncoder` — encodage spatial
- Deux couches **GATv2Conv** (attention multi-têtes sur les arêtes, avec `edge_dim` pour intégrer les features de communication).
- Connexions résiduelles + `LayerNorm` + `ELU` à chaque étage pour stabiliser l'entraînement sur des graphes de petite taille.
- Une projection linéaire (`input_proj`) sert de raccourci résiduel dès la première couche.

### 2. Pooling multi-statistique
Pour chaque graphe, l'embedding global est obtenu par concaténation de trois agrégations sur les nœuds :
- **moyenne** (`global_mean_pool`)
- **maximum** (`global_max_pool`)
- **écart-type** (implémentation batchée maison via `index_add_`, sans boucle Python)

Cela permet de capter à la fois la tendance générale et la dispersion des anomalies entre services.

### 3. Module temporel — GRU + attention
- Un **GRU bidirectionnel à 2 couches** encode la séquence d'embeddings de graphes.
- Un module **`TemporalAttentionPooling`** calcule une pondération softmax sur les pas de temps (attention additive de type Bahdanau) pour produire un résumé contextuel de la séquence.
- Le dernier état caché et le résumé attentionnel sont concaténés puis fusionnés par un MLP.

### 4. Têtes de classification (`MLPHead`)
Deux têtes indépendantes (service / panne), chacune un MLP à 2 couches cachées avec `LayerNorm`, `ReLU` et `Dropout`, partageant la même représentation fusionnée en entrée.

---

## Pipeline de données

Le dataset (`RCAGraphSequenceDataset`, dans [`dataset2.py`](./dataset2.py)) transforme trois flux CSV bruts en séquences de graphes PyTorch Geometric (`torch_geometric.data.Data`).

**Étapes principales :**

1. **Chargement et nettoyage** des trois fichiers sources (communication, ressources, événements), normalisation des noms de services.
2. **Fenêtrage temporel** (`window_sec`) : agrégation des métriques par fenêtre de temps.
3. **Features de deltas temporels** : calcul de variations (`delta_error`, `delta_latency`, `delta_throughput`, etc.) par service/paire de services au fil des fenêtres.
4. **Score de causalité par arête** : un score heuristique composite est calculé pour chaque communication inter-service, combinant :
   - la dégradation locale (erreur, latence, throughput),
   - la position relative dans la fenêtre (rang, part de la pression causale),
   - la persistance et l'accélération du signal dans le temps,
   - un score de "chemin causal" agrégeant la propagation source → destination.
5. **Construction des graphes** : un graphe par fenêtre temporelle, avec nœuds = services (features de ressources/événements/causalité) et arêtes = communications (features de communication/causalité). Les services sans communication reçoivent une auto-boucle par défaut pour rester connectés au graphe.
6. **Normalisation** : `StandardScaler` (scikit-learn) ajusté sur le train, réutilisé tel quel en validation/test.
7. **Séquençage** : regroupement des graphes en fenêtres glissantes de longueur `seq_len`, avec le label associé au **dernier pas de temps** de chaque séquence.

Le dataset gère également la **cohérence des mappings** (`service_to_idx`, `failure_to_idx`) entre train et test, afin que les classes soient encodées de façon identique lors de l'inférence.

---

## Structure du dépôt

```
.
├── model.py          # Architecture du modèle (GATEncoder, TemporalAttentionPooling, MLPHead, GATGRUMultiTask)
├── dataset2.py        # Dataset PyG : construction des graphes séquentiels + features de causalité
├── train.py           # Script d'entraînement (équilibrage de classes, AMP, checkpointing)
├── top-k.py           # Script d'évaluation détaillée (accuracy, F1, top-k, matrices de confusion)
├── tester_model/      # (attendu) CSV de validation/test au même format que l'entraînement
└── README.md
```

> Les scripts s'attendent à trouver les fichiers CSV sources dans le répertoire courant du script (`aggregated_pod_communication.csv`, `aggregated_pod_resource_consumption.csv`, `aggregated_pod_events.csv`), et un sous-dossier `tester_model/` contenant les mêmes fichiers pour la validation/le test.

---

## Installation

### Prérequis
- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric (compatible avec la version de PyTorch/CUDA installée)

### Dépendances

```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pandas numpy scikit-learn joblib
```

> ⚠️ L'installation de `torch_geometric` dépend de votre version exacte de PyTorch et de CUDA. Suivez les instructions officielles : https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Format des données d'entrée

Trois fichiers CSV sont attendus en entrée, avec un timestamp de fenêtre (`window_ts`) commun :

| Fichier | Contenu attendu |
|---|---|
| `aggregated_pod_communication.csv` | Communications entre services (source, destination, latence, débit, taux d'erreur, timestamp) |
| `aggregated_pod_resource_consumption.csv` | Consommation de ressources par service (CPU, mémoire, etc., timestamp) |
| `aggregated_pod_events.csv` | Événements Kubernetes/applicatifs par service (warnings, erreurs critiques, timestamp) |

Les labels (`y1` = service fautif, `y2` = type de panne) sont dérivés en interne par le dataset à partir de ces flux (voir `_build_window_indices` / `label_lookup` dans `dataset2.py`).

---

## Entraînement

```bash
python train.py
```

Le script `train.py` réalise :

1. **Chargement** du dataset d'entraînement avec ajustement des scalers (`fit_scaler=True`).
2. **Sauvegarde des artefacts de prétraitement** (scalers + mappings de classes) pour garantir la reproductibilité en inférence.
3. **Équilibrage des classes** sur `y1` (sous-échantillonnage plafonné à `MAX_PER_Y1` par classe) pour limiter le déséquilibre entre services.
4. **Construction d'un jeu de validation stratifié** sur `y1`, à partir d'un dossier `tester_model/` séparé.
5. **Pondération de la loss** par classe (`compute_class_weights_from_indices`) pour compenser les classes rares.
6. **Entraînement mixed-precision (AMP)** si un GPU CUDA est disponible (bf16 si supporté, sinon fp16 avec `GradScaler`).
7. **Sélection du meilleur checkpoint** sur le F1-macro de validation du service (avec la val_loss comme critère de départage).

### Sorties de l'entraînement
- `model_balanced_y1_batched.pt` — poids du meilleur modèle (`state_dict` seul)
- `best_service_f1_checkpoint.pt` — checkpoint complet (modèle, optimiseur, scaler AMP, métriques, config)
- `node_scaler.pkl`, `edge_scaler.pkl` — scalers scikit-learn
- `service_to_idx.json`, `failure_to_idx.json` — mappings classe ↔ index

### Hyperparamètres principaux (`train.py`)

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `SEQ_LEN` | 5 | Longueur de la séquence de graphes |
| `WINDOW_SEC` | 2 | Durée d'une fenêtre temporelle (s) |
| `VAL_RATIO` | 0.2 | Proportion de validation par rapport au train équilibré |
| `MAX_PER_Y1` | 700 | Plafond d'échantillons par classe de service (équilibrage) |
| `BATCH_SIZE` | 64 | Taille de batch |
| `SERVICE_LOSS_WEIGHT` | 2.5 | Pondération relative de la loss "service" vs "panne" |
| `EPOCHS` | 250 | Nombre d'époques |
| `SEED` | 42 | Graine aléatoire |

---

## Évaluation et inférence

```bash
python top-k.py
```

Ce script recharge le modèle et les artefacts de prétraitement (scalers, mappings) sauvegardés à l'entraînement, puis évalue sur le jeu de test présent dans `tester_model/`.

**Métriques calculées :**
- Accuracy et F1-macro pour chaque tâche (service, panne)
- **Top-3 / Top-5 accuracy** (utile en RCA : proposer une shortlist de causes probables plutôt qu'une seule réponse)
- Rapport de classification complet (`classification_report`) et matrices de confusion par tâche
- Export détaillé par échantillon (probabilités, top-k, prédictions) dans `predictions_debug_batched.csv`

> Le chargement du checkpoint gère automatiquement la compatibilité `weights_only` introduite dans PyTorch 2.6+, avec repli sécurisé si nécessaire.

---

## Artefacts produits

| Fichier | Description |
|---|---|
| `best_service_f1_checkpoint.pt` | Checkpoint complet à utiliser pour l'inférence |
| `model_balanced_y1_batched.pt` | Poids seuls du meilleur modèle |
| `node_scaler.pkl` / `edge_scaler.pkl` | Normalisation à réappliquer sur toute nouvelle donnée |
| `service_to_idx.json` / `failure_to_idx.json` | Mappings de classes, indispensables pour interpréter les prédictions |
| `predictions_debug_batched.csv` | Prédictions détaillées avec top-k et probabilités |

---




## Licence

.....
