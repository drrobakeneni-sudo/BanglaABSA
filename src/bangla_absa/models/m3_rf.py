from bangla_absa.models.baselines.tfidf import TfidfModel


class RandomForestModel(TfidfModel):
    """Bootstrap forest consuming sparse features without dense conversion."""
