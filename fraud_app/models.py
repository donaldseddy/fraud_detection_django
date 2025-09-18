from django.db import models

class User(models.Model):
    """
    Représente un utilisateur (normal ou fraudeur).
    """
    full_name = models.CharField(max_length=200)
    email_address = models.EmailField(unique=True)
    address = models.CharField(max_length=255)
    country = models.CharField(max_length=100)
    date_created = models.DateField(auto_now_add=True)

    credit_card_number = models.CharField(max_length=25, null=True, blank=True)
    is_chargemap_pro_user = models.BooleanField(default=False)

    invoices_count = models.IntegerField(default=0)
    paid_invoices_count = models.IntegerField(default=0)
    current_debt_amount = models.FloatField(default=0.0)

    feedback_count = models.IntegerField(default=0)
    payment_mean = models.CharField(max_length=50, null=True, blank=True)
    payment_time_avg = models.FloatField(default=0.0)  # en jours

    def __str__(self):
        return f"{self.full_name} ({'Fraud?' if self.current_debt_amount > 0 else 'OK'})"


class Prediction(models.Model):
    """
    Stocke les résultats du modèle ML pour chaque utilisateur.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    is_fraud = models.BooleanField()
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.user.full_name}: {'Fraud' if self.is_fraud else 'Normal'}"
