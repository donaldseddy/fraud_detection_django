import joblib
import pandas as pd
from django.core.management.base import BaseCommand
from fraud_app.models import User, Prediction

class Command(BaseCommand):
    help = "Applique le modèle ML aux utilisateurs et stocke les prédictions"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            type=str,
            default="ml/model.pkl",
            help="Chemin vers le modèle entraîné"
        )

    def handle(self, *args, **options):
        model_path = options["model"]
        self.stdout.write(self.style.NOTICE(f"📂 Chargement du modèle {model_path}..."))
        bundle = joblib.load(model_path)

        model = bundle["model"]
        preprocessor = bundle["preprocessor"]

        # Récupération des users depuis PostgreSQL
        users = User.objects.all()
        if not users.exists():
            self.stdout.write(self.style.ERROR("❌ Aucun utilisateur trouvé en base."))
            return

        # Conversion des users en DataFrame
        df = pd.DataFrame(list(users.values()))

        # 🔹 Features attendues par le modèle
        features = [
            "is_chargemap_pro_user", "feedback_count",
            "invoices_count", "paid_invoices_count",
            "current_debt_amount", "payment_time_avg",
            "address", "country", "payment_mean"
        ]

        # Vérification que toutes les colonnes existent
        for col in features:
            if col not in df.columns:
                df[col] = None  # créer la colonne si absente

        # 🔹 Gestion des valeurs manquantes
        df["address"] = df["address"].fillna("Unknown")
        df["country"] = df["country"].fillna("Unknown")
        df["payment_mean"] = df["payment_mean"].fillna("Unknown")

        df["feedback_count"] = df["feedback_count"].fillna(0)
        df["invoices_count"] = df["invoices_count"].fillna(0)
        df["paid_invoices_count"] = df["paid_invoices_count"].fillna(0)
        df["current_debt_amount"] = df["current_debt_amount"].fillna(0.0)
        df["payment_time_avg"] = df["payment_time_avg"].fillna(0.0)

        # Préparation des features pour prédiction
        X = preprocessor.transform(df[features])
        y_proba = model.predict_proba(X)[:, 1]

        predictions_created = 0
        for user, prob in zip(users, y_proba):
            is_fraud = prob > 0.5
            Prediction.objects.create(
                user=user,
                is_fraud=is_fraud,
                probability=float(prob)
            )
            predictions_created += 1

        self.stdout.write(self.style.SUCCESS(
            f"{predictions_created} prédictions enregistrées dans la base."
        ))
