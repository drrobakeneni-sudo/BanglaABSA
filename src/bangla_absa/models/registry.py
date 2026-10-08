from bangla_absa.models.baselines.aen import AEN
from bangla_absa.models.baselines.atae_lstm import ATAELSTM
from bangla_absa.models.baselines.fasttext_bilstm import FastTextBiLSTM
from bangla_absa.models.baselines.ian import IAN
from bangla_absa.models.baselines.lcf_bert import LCFBERT
from bangla_absa.models.baselines.memnet import MemNet
from bangla_absa.models.baselines.pair_qa_m import PairQAM
from bangla_absa.models.baselines.plm import PLM, ThreeHeads
from bangla_absa.models.baselines.ram import RAM
from bangla_absa.models.baselines.spc import SPC
from bangla_absa.models.baselines.tfidf import TfidfModel
from bangla_absa.models.m1_lr import LogisticRegressionModel
from bangla_absa.models.m2_xgb import XGBoostModel
from bangla_absa.models.m3_rf import RandomForestModel
from bangla_absa.models.m4_absa_net import ABSANet
from bangla_absa.models.m5_transnet import TransNet
from bangla_absa.models.m6_ensemble import Stacker

CLASSICAL = {
    "m1": LogisticRegressionModel,
    "m2": XGBoostModel,
    "m3": RandomForestModel,
    "naive_bayes": TfidfModel,
    "svm": TfidfModel,
}
NEURAL = {
    "m4": ABSANet,
    "m5": TransNet,
    "fasttext_bilstm": FastTextBiLSTM,
    "memnet": MemNet,
    "atae_lstm": ATAELSTM,
    "ian": IAN,
    "ram": RAM,
    "mbert": PLM,
    "xlmr_base": PLM,
    "mdeberta": PLM,
    "xlmr_large": PLM,
    "banglishbert": PLM,
    "banglabert": PLM,
    "banglabert_large": PLM,
    "three_heads": ThreeHeads,
    "spc": SPC,
    "pair_qa_m": PairQAM,
    "aen": AEN,
    "lcf_bert": LCFBERT,
}
CONTROLS = ("m5_five_seed_vote", "equal_vote", "logistic_stacking")


def build_model(cfg, seed=17, encoder=None):
    if cfg.name in CLASSICAL:
        return CLASSICAL[cfg.name](cfg, seed)
    if cfg.name == "m6":
        return Stacker(cfg.dropout)
    if cfg.name not in NEURAL:
        raise ValueError(f"Unknown model {cfg.name}; select a registered name")
    return NEURAL[cfg.name](cfg, encoder)
