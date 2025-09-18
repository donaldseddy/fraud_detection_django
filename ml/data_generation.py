import pandas as pd
import random
from faker import Faker
import argparse
import os

fake = Faker()

def generate_users(n_users=1000, fraud_ratio=0.1):
    """
    Génère un dataset synthétique avec fraudeurs et utilisateurs normaux.
    """
    users = []
    n_fraud = int(n_users * fraud_ratio)
    n_normal = n_users - n_fraud

    # --- Fraudeurs ---
    shared_cards = [fake.credit_card_number() for _ in range(max(1, n_fraud // 5))]
    shared_addresses = [fake.address() for _ in range(max(1, n_fraud // 5))]

    for i in range(n_fraud):
        card = random.choice(shared_cards)
        addr = random.choice(shared_addresses)
        email_base = fake.last_name().lower()

        users.append({
            "full_name": fake.name(),
            "email_address": f"{email_base}{random.randint(100,999)}@fraudmail.com",
            "address": addr,
            "country": fake.country(),
            "credit_card_number": card,
            "is_chargemap_pro_user": True,
            "invoices_count": random.randint(1, 20),
            "paid_invoices_count": random.randint(0, 5),
            "current_debt_amount": round(random.uniform(100, 5000), 2),
            "feedback_count": random.randint(0, 1),
            "payment_mean": random.choice(["Card", "Paypal", "Crypto"]),
            "payment_time_avg": random.uniform(15, 60),
            "is_fraud": 1
        })

    # --- Normaux ---
    for i in range(n_normal):
        users.append({
            "full_name": fake.name(),
            "email_address": fake.email(),
            "address": fake.address(),
            "country": fake.country(),
            "credit_card_number": fake.credit_card_number(),
            "is_chargemap_pro_user": False,
            "invoices_count": random.randint(5, 50),
            "paid_invoices_count": random.randint(5, 50),
            "current_debt_amount": 0.0,
            "feedback_count": random.randint(1, 10),
            "payment_mean": random.choice(["Card", "Paypal"]),
            "payment_time_avg": random.uniform(1, 15),
            "is_fraud": 0
        })

    return pd.DataFrame(users)


def save_dataset(df, output):
    """Sauvegarde un dataset en CSV"""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f" Dataset généré : {output} ({len(df)} utilisateurs, {df['is_fraud'].mean()*100:.1f}% fraudeurs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="Nombre d'utilisateurs")
    parser.add_argument("--fraud_ratio", type=float, default=0.1, help="Proportion de fraudeurs")
    parser.add_argument("--output", type=str, help="Nom du fichier CSV de sortie")
    parser.add_argument("--batch", action="store_true", help="Générer plusieurs datasets prédéfinis")

    args = parser.parse_args()

    if args.batch:
        # Génération de plusieurs datasets
        configs = [
            (1000, 0.1, "data/users_1k.csv"),
            (10000, 0.05, "data/users_10k.csv"),
        ]
        for n, ratio, path in configs:
            df = generate_users(n, ratio)
            save_dataset(df, path)
    else:
        # Génération simple
        if not args.n or not args.output:
            print(" Vous devez spécifier --n et --output (sauf si vous utilisez --batch).")
        else:
            df = generate_users(args.n, args.fraud_ratio)
            save_dataset(df, args.output)
