import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from ml.preprocessing import load_dataset, preprocess_data


def train_and_evaluate(dataset="data/users_10k.csv", model_path="ml/model.pkl", optimize=True):
    # Charger dataset
    df = load_dataset(dataset)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Modèle de base
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

    if optimize:
        # Hyperparamètres à explorer
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=20,                # nombre d’essais (équilibre temps/perf)
            cv=3,                     # cross-validation à 3 folds
            scoring="roc_auc",        # on optimise le ROC-AUC
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        print(" Optimisation des hyperparamètres...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(" Best params:", search.best_params_)

    else:
        model = base_model
        model.fit(X_train, y_train)

    #  Évaluation finale
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("✅ Rapport de classification :\n", classification_report(y_test, y_pred))
    print("✅ ROC-AUC :", roc_auc_score(y_test, y_proba))

    # 💾 Sauvegarder modèle + preprocessing
    joblib.dump({"model": model, "preprocessor": preprocessor}, model_path)
    print(f"💾 Modèle sauvegardé sous {model_path}")

    return model, preprocessor


if __name__ == "__main__":
    train_and_evaluate()
