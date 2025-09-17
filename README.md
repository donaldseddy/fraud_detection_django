# Fraud Detection System

Projet de détection de fraude développé en Django, PostgreSQL et Machine Learning.
Il permet de prédire si un utilisateur est susceptible d’avoir des dettes impayées
à partir de ses données de facturation et comportementales.


## fraud_detection_django
La problématique est de détecter les fraudeurs en simulant des comportements crédibles puis en construisant un pipeline ML robuste et interprétable.Nous allons le depployer en  Dev & Prod.  
La solution structurée suit un workflow complet : Data → Features → Modèle → Évaluation → Interprétation → Livrables



## 📂 Structure du projet

fraud_detection_django/
│── fraud_detection/        # Configuration Django
│── fraud_app/              # Application métier
│── ml/                     # Scripts Machine Learning
│── data/                   # Données brutes et traitées
│── requirements.txt        # Dépendances Python
│── manage.py               # Entrée du projet
│── env                     # parametre des variables env (local & prod)


## Installation & Configuration

### Prérequis
    - Python 3.9+
    - PostgreSQL 13+
    - pip / virtualenv

### Installations variables d'environnement
    - usage de la biblio python-decouple
    - 
    
### **Configuration base de données**