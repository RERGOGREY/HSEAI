from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    unique_values = np.unique(feature_vector)
    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    ginis = np.zeros_like(thresholds)

    total_samples = len(target_vector)
    total_p1 = np.sum(target_vector == 1)
    total_p0 = total_samples - total_p1

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    left_p1, left_p0 = 0, 0
    right_p1, right_p0 = total_p1, total_p0

    for i, threshold in enumerate(thresholds):
        mask = sorted_features <= threshold
        left_size = np.sum(mask)
        right_size = total_samples - left_size

        left_p1 = np.sum(sorted_targets[:left_size] == 1)
        left_p0 = left_size - left_p1
        right_p1 = total_p1 - left_p1
        right_p0 = total_p0 - left_p0

        h_left = 1 - (left_p1 / left_size) ** 2 - (left_p0 / left_size) ** 2 if left_size > 0 else 0
        h_right = 1 - (right_p1 / right_size) ** 2 - (right_p0 / right_size) ** 2 if right_size > 0 else 0

        ginis[i] = - (left_size / total_samples) * h_left - (right_size / total_samples) * h_right

    best_idx = np.argmax(ginis)
    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if not all(x in {"real", "categorical"} for x in feature_types):
            raise ValueError("There is an unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]) or (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y.astype(bool), feature].ravel() if isinstance(sub_X, np.ndarray) else sub_X.iloc[sub_y.astype(bool).values, feature])  
                ratio = {key: counts[key] / (clicks[key] + 1) for key in counts}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            if len(feature_vector) < self._min_samples_split:
                continue

            result = find_best_split(feature_vector, sub_y)
            if result is not None:
                _, _, threshold, gini = result
                if gini is not None and (gini_best is None or gini > gini_best):
                    feature_best, gini_best, threshold_best = feature, gini, threshold
                    if threshold is not None:
                        split = feature_vector < threshold 
                    else:
                        split = np.zeros_like(feature_vector, dtype=bool)

        if feature_best is None or gini_best is None or len(sub_y) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best if self._feature_types[feature_best] == "real" else None
        node["categories_split"] = sorted_categories if self._feature_types[feature_best] == "categorical" else None

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_index = int(node["feature_split"])
        feature_value = x[feature_index]

        if self._feature_types[feature_index] == "real":
            return self._predict_node(x, node["left_child"] if feature_value < node["threshold"] else node["right_child"])
        elif self._feature_types[feature_index] == "categorical":
            if node["categories_split"] is not None and feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError(f"Unknown feature type: {self._feature_types[feature_index]} for feature {feature_index}")

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if len(self._feature_types) != X.shape[1]:
            raise ValueError("Feature types length does not match number of features in X")
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_node(x, self._tree) for x in X])

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
