from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from bangla_absa.data.schema import labels_array
from bangla_absa.io import read_json, write_json


class TfidfModel:
    def __init__(self, cfg, seed=17):
        self.cfg, self.seed = cfg, seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=cfg.max_features,
            token_pattern=r"(?u)[\w\u0980-\u09ff]+|[^\w\s]",
        )
        self.estimators, self.class_maps = [], []

    def fit(self, train, validation=None, role="train"):
        if role != "train":
            raise ValueError("Classical models may only fit on train")
        x = self.vectorizer.fit_transform([r["text"] for r in train])
        y = labels_array(train)
        xv = self.vectorizer.transform([r["text"] for r in validation]) if validation else None
        yv = labels_array(validation) if validation else None
        self.estimators, self.class_maps = [], []
        for a in range(3):
            valid = y[:, a] >= 0
            classes, counts = np.unique(y[valid, a], return_counts=True)
            if len(classes) == 0:
                raise ValueError("Every aspect needs at least one observed training label")
            name = self.cfg.name
            if len(classes) == 1:
                estimator = DummyClassifier(strategy="constant", constant=classes[0])
            elif name == "m1":
                estimator = LogisticRegression(
                    C=1, solver="lbfgs", max_iter=1000, random_state=self.seed
                )
            elif name == "m2":
                from xgboost import XGBClassifier

                if validation is None:
                    raise ValueError("M2 requires validation for early stopping; never supply test")
                estimator = XGBClassifier(
                    tree_method="hist",
                    learning_rate=0.080,
                    n_estimators=self.cfg.trees or 500,
                    reg_lambda=1.0,
                    max_depth=6,
                    objective="multi:softprob",
                    num_class=len(classes),
                    eval_metric="mlogloss",
                    early_stopping_rounds=25,
                    random_state=self.seed,
                    n_jobs=2,
                )
            elif name == "m3":
                estimator = RandomForestClassifier(
                    n_estimators=self.cfg.trees or 800,
                    bootstrap=True,
                    max_features="sqrt",
                    random_state=self.seed,
                    n_jobs=2,
                )
            elif name == "naive_bayes":
                estimator = MultinomialNB(alpha=1)
            elif name == "svm":
                folds = min(self.cfg.svm_calibration_folds, int(counts.min()))
                if folds < 2:
                    raise ValueError("SVM calibration needs at least two train examples per class")
                estimator = CalibratedClassifierCV(
                    LinearSVC(C=1, random_state=self.seed), method="sigmoid", cv=folds
                )
            else:
                raise ValueError("Unknown TF-IDF model")
            if name == "m2" and len(classes) > 1:
                mapping = {int(c): i for i, c in enumerate(classes)}
                v = np.isin(yv[:, a], classes)
                if not v.any():
                    raise ValueError("M2 validation has no labels among observed training classes")
                estimator.fit(
                    x[valid],
                    np.array([mapping[int(c)] for c in y[valid, a]]),
                    eval_set=[(xv[v], np.array([mapping[int(c)] for c in yv[v, a]]))],
                    verbose=False,
                )
                self.class_maps.append(classes.tolist())
            else:
                estimator.fit(x[valid], y[valid, a])
                self.class_maps.append([int(c) for c in estimator.classes_])
            self.estimators.append(estimator)
        return self

    def predict_proba(self, records):
        x = self.vectorizer.transform([r["text"] for r in records])
        result = np.zeros((len(records), 3, 3))
        for a, estimator in enumerate(self.estimators):
            result[:, a, self.class_maps[a]] = estimator.predict_proba(x)
        return result

    def save(self, directory):
        import skops.io as sio

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sio.dump(self.vectorizer, directory / "vectorizer.skops")
        formats = []
        for a, estimator in enumerate(self.estimators):
            if isinstance(estimator, RandomForestClassifier):
                from bangla_absa.models.baselines.forest_io import PortableForest

                write_json(
                    directory / f"aspect-{a}.json", PortableForest.from_sklearn(estimator).trees
                )
                formats.append("forest_json")
            elif type(estimator).__module__.startswith("xgboost"):
                estimator.save_model(directory / f"aspect-{a}.json")
                formats.append("xgboost")
            else:
                sio.dump(estimator, directory / f"aspect-{a}.skops")
                formats.append("skops")
        write_json(directory / "classical.json", {"formats": formats, "classes": self.class_maps})

    def load(self, directory):
        import skops.io as sio
        from xgboost import XGBClassifier

        directory = Path(directory)
        # No automatic trust of unknown types or pickle/joblib deserialization.
        self.vectorizer = sio.load(directory / "vectorizer.skops", trusted=[])
        metadata = read_json(directory / "classical.json")
        self.estimators, self.class_maps = [], metadata["classes"]
        for a, format_name in enumerate(metadata["formats"]):
            if format_name == "xgboost":
                model = XGBClassifier()
                model.load_model(directory / f"aspect-{a}.json")
            elif format_name == "forest_json":
                from bangla_absa.models.baselines.forest_io import PortableForest

                model = PortableForest(read_json(directory / f"aspect-{a}.json"))
            elif format_name == "skops":
                model = sio.load(
                    directory / f"aspect-{a}.skops",
                    trusted=[
                        "sklearn.calibration._CalibratedClassifier",
                        "sklearn.calibration._SigmoidCalibration",
                    ],
                )
            else:
                raise ValueError("Unsupported estimator serialization")
            self.estimators.append(model)
        return self
