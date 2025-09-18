import joblib
import shap
from ml.preprocessing import load_dataset, preprocess_data

def explain_model(dataset="data/users_1k.csv", model_path="ml/model.pkl"):
    # Charger modèle + preprocess
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    # Charger données
    df = load_dataset(dataset)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Afficher importance globale
    shap.summary_plot(shap_values, X_test, show=True)

if __name__ == "__main__":
    explain_model()
