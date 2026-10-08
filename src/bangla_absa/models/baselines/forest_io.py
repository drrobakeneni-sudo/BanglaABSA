"""Portable tree inference without deserializing native sklearn Tree objects."""

import numpy as np


class PortableForest:
    def __init__(self, trees):
        self.trees = trees
        if not trees:
            raise ValueError("Empty forest")
        for tree in trees:
            size = len(tree["feature"])
            if any(len(tree[key]) != size for key in ("left", "right", "threshold", "value")):
                raise ValueError("Inconsistent tree arrays")
            for node, (left, right) in enumerate(zip(tree["left"], tree["right"])):
                if (left == -1) != (right == -1):
                    raise ValueError("Malformed tree leaf")
                if left != -1 and not (node < left < size and node < right < size):
                    raise ValueError("Invalid or cyclic tree child indices")
            values = np.asarray(tree["value"], dtype=float)
            if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid tree class counts")
            if (values.sum(-1) <= 0).any():
                raise ValueError("Empty tree distribution")

    @classmethod
    def from_sklearn(cls, forest):
        trees = []
        for estimator in forest.estimators_:
            t = estimator.tree_
            trees.append(
                {
                    "feature": t.feature.tolist(),
                    "left": t.children_left.tolist(),
                    "right": t.children_right.tolist(),
                    "threshold": t.threshold.tolist(),
                    "value": t.value[:, 0, :].tolist(),
                }
            )
        return cls(trees)

    def predict_proba(self, matrix):
        result = np.zeros((matrix.shape[0], len(self.trees[0]["value"][0])))
        matrix = matrix.tocsr()
        for tree in self.trees:
            for i in range(matrix.shape[0]):
                node = 0
                while tree["left"][node] != -1:
                    feature = tree["feature"][node]
                    if not 0 <= feature < matrix.shape[1]:
                        raise ValueError("Tree feature index outside input width")
                    # sklearn trees compare float32 features.
                    value = np.float32(matrix[i, feature])
                    node = (
                        tree["left"][node]
                        if value <= tree["threshold"][node]
                        else tree["right"][node]
                    )
                counts = np.asarray(tree["value"][node])
                result[i] += counts / counts.sum()
        return result / len(self.trees)
