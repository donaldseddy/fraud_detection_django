# Fraud Detection System

Projet : détection de fraude (test technique) — stack : Django + PostgreSQL + scikit-learn
Objectif : générer un dataset cohérent, entraîner un modèle de classification pour détecter des utilisateurs susceptibles d’avoir des dettes impayées, exposer les prédictions via Django et enregistrer les résultats en base.


## fraud_detection_django
La problématique est de détecter les fraudeurs en simulant des comportements crédibles puis en construisant un pipeline ML robuste et interprétable.Nous allons le depployer en  Dev & Prod.  
La solution structurée suit un workflow complet : Data → Features → Modèle → Évaluation → Interprétation → Livrables



##  Structure du projet

fraud_detection_django/
│── fraud_detection/            # Config Django (settings, urls, wsgi, asgi)
│── fraud_app/                  # Application principale
|   ├── management/commands/
│   │   ├── load_users.py
│   │   └── predict_users.py
│   │── models.py               # Définition des modèles (User, Prediction)
│   │── views.py                # API (prédiction, export CSV)
│   │── urls.py                 # Routage API
│   │── admin.py                # Intégration dans l'admin Django
│   │── serializers.py          # (Optionnel) API REST DRF
│   │── migrations/             # Migrations Django
│
│── ml/                         # Partie Machine Learning
│   │── data_generation.py      # Génération de données synthétiques
│   │── preprocessing.py        # Nettoyage & Feature engineering
│   │── modeling.py             # Entraînement, évaluation, sauvegarde modèle
│   │── explainability.py       # SHAP, Feature importance
│   │── model.pkl               # Modèle sauvegardé
│
│── data/                       # Datasets générés
│   │── users_1k.csv
│   │── users_10k.csv
│
│── manage.py                   # Entrée du projet Django
│── requirements.txt            # Dépendances
│── README.md                   # Documentation
│── .env                        # Variables d'environnement
│── env /
    │──.env.local               # Variables dev
    │──.env.production          # Variables prod


## Etapes & Configuration

### Prérequis
    - Python 3.9+
    - PostgreSQL 13+
    - pip / virtualenv

### Installations variables d'environnement
    - usage de la biblio python-decouple pour la cohabitation des variable d'env dev et prod
    

### **Configuration base de données**
    - Implementation des models
    - Model User Représente un utilisateur (normal ou fraudeur)
    - Model Prediction pour Stocke les résultats du modèle ML pour chaque utilisateur

### **Configuration Model ML**
    - genereration hazardeuse mais coherente des utilisateur grace a data_generation.py en introduisant une proportion de fraudeurs ici elle est 10% pour 1000 et 5% pour 10K
    - 