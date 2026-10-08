from bangla_absa.models.baselines.tfidf import TfidfModel


class XGBoostModel(TfidfModel):
    """Histogram gradient boosting, validation-only stopping, explicit class mapping."""
