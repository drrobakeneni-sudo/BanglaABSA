from bangla_absa.models.baselines.tfidf import TfidfModel


class LogisticRegressionModel(TfidfModel):
    """Three masked LBFGS estimators on train-fitted sparse TF-IDF 1–3 grams."""
