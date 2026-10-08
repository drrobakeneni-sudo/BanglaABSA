from bangla_absa.models.baselines.spc import SPC


class PairQAM(SPC):
    """QA-M sentence pair: category question and 3-class answer (no binary expansion).

    The tokenizer uses a question rather than the SPC category phrase. Shared
    pair-classification computation follows the auxiliary-sentence formulation.
    """
