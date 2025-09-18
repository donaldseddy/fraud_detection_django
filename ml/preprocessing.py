import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_dataset(path="data/users_1k.csv"):
    """
    Charge le dataset généré (CSV).
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Applique le prétraitement :
    - séparation X / y
    - encodage des variables catégorielles
    - scaling des variables numériques
    - retourne train/test prêts pour le modèle
    """

    # cible
    y = df["is_fraud"]

    #  features (on enlève les colonnes inutiles pour l’entraînement)
    X = df.drop(columns=["is_fraud", "full_name", "email_address", "credit_card_number"])

    # Colonnes numériques
    numeric_features = ["invoices_count", "paid_invoices_count", "current_debt_amount", "feedback_count", "payment_time_avg"]

    # Colonnes catégorielles
    categorical_features = ["address", "country", "payment_mean", "is_chargemap_pro_user"]

    # Pipelines de transformation
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Transformer combiné
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit-transform sur train, transform sur test
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor


if __name__ == "__main__":
    df = load_dataset("data/users_1k.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print("Prétraitement terminé :")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
