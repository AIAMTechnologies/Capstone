"""Train a lead conversion classifier with preprocessing and evaluation.

The script loads the exported lead dataset, handles preprocessing (missing
values, encoding, scaling), performs a stratified train/validation split,
trains an ensemble model, evaluates it, and persists both the trained pipeline
and metrics.

When scikit-learn is available the script uses its preprocessing and
``RandomForestClassifier`` implementation. In restricted environments without
those dependencies it falls back to a lightweight, pure-Python approximation.
Regardless of the path taken, the persisted artifact mirrors a scikit-learn
style pipeline with preprocessing followed by a random forest classifier. The
artifact is saved with the ``.joblib`` extension so it can be loaded with
``joblib.load`` in environments where scikit-learn is available.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - fallback when joblib is unavailable
    class _JoblibFallback:
        @staticmethod
        def dump(obj, filename):
            with open(filename, "wb") as fh:
                pickle.dump(obj, fh)

        @staticmethod
        def load(filename):
            with open(filename, "rb") as fh:
                return pickle.load(fh)

    joblib = _JoblibFallback()  # type: ignore


try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
    from sklearn.compose import ColumnTransformer as SKColumnTransformer  # type: ignore
    from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier  # type: ignore
    from sklearn.impute import SimpleImputer as SKSimpleImputer  # type: ignore
    from sklearn.model_selection import train_test_split as sk_train_test_split  # type: ignore
    from sklearn.pipeline import Pipeline as SKPipeline  # type: ignore
    from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder, StandardScaler as SKStandardScaler  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when scikit-learn is unavailable
    pd = None
    SKColumnTransformer = None
    SKRandomForestClassifier = None
    SKSimpleImputer = None
    sk_train_test_split = None
    SKPipeline = None
    SKOneHotEncoder = None
    SKStandardScaler = None
    SKLEARN_AVAILABLE = False


NUMERIC_COLUMNS = [
    "estimated_project_value",
    "response_time_hours",
    "num_followups",
    "installer_distance_km",
]
CATEGORICAL_COLUMNS = ["city", "province", "job_type", "lead_source"]
TARGET_COLUMN = "converted"


def _safe_float(value: str) -> Optional[float]:
    try:
        stripped = value.strip()
    except AttributeError:
        return None
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _safe_str(value: str) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


@dataclass
class Preprocessor:
    """Handle imputation, scaling, and one-hot encoding."""

    numeric_stats: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    categorical_values: Dict[str, List[str]] = field(default_factory=dict)
    categorical_modes: Dict[str, str] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def fit(self, records: Sequence[Dict[str, object]]) -> None:
        self.numeric_stats.clear()
        self.categorical_values.clear()
        self.categorical_modes.clear()

        for column in NUMERIC_COLUMNS:
            values = [rec[column] for rec in records if isinstance(rec[column], (int, float))]
            if not values:
                median = mean = 0.0
                std = 1.0
            else:
                sorted_vals = sorted(float(v) for v in values)
                n = len(sorted_vals)
                mid = n // 2
                if n % 2 == 0:
                    median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
                else:
                    median = sorted_vals[mid]
                mean = sum(sorted_vals) / n
                variance = sum((val - mean) ** 2 for val in sorted_vals) / n
                std = math.sqrt(variance) or 1.0
            self.numeric_stats[column] = (median, mean, std)

        for column in CATEGORICAL_COLUMNS:
            values = [rec[column] for rec in records if isinstance(rec[column], str) and rec[column]]
            if not values:
                mode = "missing"
                categories: List[str] = [mode]
            else:
                counts = Counter(values)
                mode = counts.most_common(1)[0][0]
                categories = sorted(set(values))
            self.categorical_modes[column] = mode
            self.categorical_values[column] = categories

        feature_names: List[str] = []
        feature_names.extend(f"num::{col}" for col in NUMERIC_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            for category in self.categorical_values[column]:
                feature_names.append(f"cat::{column}::{category}")
        self.feature_names = feature_names

    def transform(self, records: Sequence[Dict[str, object]]) -> List[List[float]]:
        matrix: List[List[float]] = []
        for rec in records:
            row: List[float] = []
            for column in NUMERIC_COLUMNS:
                median, mean, std = self.numeric_stats[column]
                value = rec[column]
                numeric = float(value) if isinstance(value, (int, float)) else median
                row.append((numeric - mean) / std)
            for column in CATEGORICAL_COLUMNS:
                categories = self.categorical_values[column]
                mode = self.categorical_modes[column]
                value = rec[column] if isinstance(rec[column], str) and rec[column] else mode
                for category in categories:
                    row.append(1.0 if value == category else 0.0)
            matrix.append(row)
        return matrix


@dataclass
class TreeNode:
    prediction: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def is_leaf(self) -> bool:
        return self.feature_index is None or self.left is None or self.right is None


class SimpleRandomForestClassifier:
    """A lightweight random forest implementation for binary classification."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self.trees: List[TreeNode] = []

    @staticmethod
    def _gini(labels: Sequence[int]) -> float:
        if not labels:
            return 0.0
        total = len(labels)
        positives = sum(labels)
        negatives = total - positives
        p_pos = positives / total
        p_neg = negatives / total
        return 1.0 - (p_pos ** 2 + p_neg ** 2)

    def _best_split(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        feature_indices: Sequence[int],
    ) -> Tuple[Optional[int], Optional[float], float]:
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_gain = 0.0
        parent_gini = self._gini(y)

        for feature in feature_indices:
            values = sorted(set(row[feature] for row in X))
            if len(values) <= 1:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left_labels = [label for row, label in zip(X, y) if row[feature] <= threshold]
                right_labels = [label for row, label in zip(X, y) if row[feature] > threshold]
                if len(left_labels) < self.min_samples_split or len(right_labels) < self.min_samples_split:
                    continue
                left_gini = self._gini(left_labels)
                right_gini = self._gini(right_labels)
                total = len(y)
                weighted_gini = (len(left_labels) / total) * left_gini + (len(right_labels) / total) * right_gini
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        depth: int,
    ) -> TreeNode:
        prediction = sum(y) / len(y)
        node = TreeNode(prediction=prediction)
        if depth >= self.max_depth or len(set(y)) == 1:
            return node

        n_features = len(X[0])
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        feature_indices = self._rng.sample(range(n_features), k=min(max_features, n_features))

        feature, threshold, gain = self._best_split(X, y, feature_indices)
        if feature is None or threshold is None or gain <= 1e-9:
            return node

        left_X: List[List[float]] = []
        left_y: List[int] = []
        right_X: List[List[float]] = []
        right_y: List[int] = []
        for row, label in zip(X, y):
            if row[feature] <= threshold:
                left_X.append(list(row))
                left_y.append(label)
            else:
                right_X.append(list(row))
                right_y.append(label)

        if not left_X or not right_X:
            return node

        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        return node

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        self.trees = []
        n_samples = len(X)
        for _ in range(self.n_estimators):
            indices = [self._rng.randrange(n_samples) for _ in range(n_samples)]
            sample_X = [X[i] for i in indices]
            sample_y = [y[i] for i in indices]
            tree = self._build_tree(sample_X, sample_y, depth=0)
            self.trees.append(tree)

    def _predict_tree(self, tree: TreeNode, row: Sequence[float]) -> float:
        node = tree
        while not node.is_leaf():
            assert node.feature_index is not None and node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
            if node is None:
                break
        return 0.5 if node is None else node.prediction

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        probabilities: List[Tuple[float, float]] = []
        for row in X:
            tree_probs = [self._predict_tree(tree, row) for tree in self.trees]
            prob_pos = sum(tree_probs) / len(tree_probs) if tree_probs else 0.5
            prob_pos = min(max(prob_pos, 1e-6), 1 - 1e-6)
            probabilities.append((1 - prob_pos, prob_pos))
        return probabilities

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        return [1 if prob_pos >= 0.5 else 0 for _, prob_pos in self.predict_proba(X)]


def load_dataset(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    records: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record: Dict[str, object] = {}
            for column in NUMERIC_COLUMNS:
                record[column] = _safe_float(row.get(column, ""))
            for column in CATEGORICAL_COLUMNS:
                record[column] = _safe_str(row.get(column, ""))
            target_value = row.get(TARGET_COLUMN, "0")
            record[TARGET_COLUMN] = int(float(target_value))
            records.append(record)
    return records


def write_base64_file(source: Path, destination: Path) -> None:
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    chunk_size = 76
    wrapped = "\n".join(
        encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)
    )
    destination.write_text(wrapped + "\n")


def stratified_split(
    records: Sequence[Dict[str, object]],
    test_size: float,
    random_state: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(random_state)
    positives = [rec for rec in records if rec[TARGET_COLUMN] == 1]
    negatives = [rec for rec in records if rec[TARGET_COLUMN] == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_test = max(1, int(round(len(positives) * test_size))) if positives else 0
    n_neg_test = max(1, int(round(len(negatives) * test_size))) if negatives else 0

    test = positives[:n_pos_test] + negatives[:n_neg_test]
    train = positives[n_pos_test:] + negatives[n_neg_test:]

    if not train or not test:
        raise ValueError("Not enough data to create train/validation splits")

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> List[List[int]]:
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    return [[tn, fp], [fn, tp]]


def roc_auc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    positives = [score for score, target in zip(y_scores, y_true) if target == 1]
    negatives = [score for score, target in zip(y_scores, y_true) if target == 0]
    if not positives or not negatives:
        return 0.5
    total = 0.0
    for ps in positives:
        for ns in negatives:
            if ps > ns:
                total += 1.0
            elif ps == ns:
                total += 0.5
    return total / (len(positives) * len(negatives))


def classification_report(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    report_lines = [
        "precision    recall  f1-score   support",
    ]
    for label, support in ((0, tn + fp), (1, tp + fn)):
        if support == 0:
            precision = recall = f1 = 0.0
        elif label == 1:
            precision, recall, f1 = precision_recall_f1(y_true, y_pred)
        else:
            tn_precision = tn / (tn + fn) if tn + fn else 0.0
            tn_recall = tn / (tn + fp) if tn + fp else 0.0
            if tn_precision + tn_recall:
                tn_f1 = 2 * tn_precision * tn_recall / (tn_precision + tn_recall)
            else:
                tn_f1 = 0.0
            precision, recall, f1 = tn_precision, tn_recall, tn_f1
        report_lines.append(
            f"{label:>5} {precision:>10.2f} {recall:>8.2f} {f1:>9.2f} {support:>10}"
        )
    return "\n".join(report_lines)


def evaluate_model(y_true: Sequence[int], probabilities: Sequence[float]) -> Dict[str, object]:
    y_pred = [1 if prob >= 0.5 else 0 for prob in probabilities]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics


def train(
    dataset_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, object]:
    records = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lead_conversion_model.joblib"
    model_base64_path = output_dir / "lead_conversion_model.joblib.base64"
    metrics_path = output_dir / "lead_conversion_metrics.json"

    if SKLEARN_AVAILABLE and pd is not None and sk_train_test_split is not None:
        df = pd.DataFrame(records)
        X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        y = df[TARGET_COLUMN]

        X_train, X_val, y_train, y_val = sk_train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        numeric_transformer = SKPipeline(
            steps=[
                ("imputer", SKSimpleImputer(strategy="median")),
                ("scaler", SKStandardScaler()),
            ]
        )
        categorical_transformer = SKPipeline(
            steps=[
                ("imputer", SKSimpleImputer(strategy="most_frequent")),
                ("encoder", SKOneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = SKColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_COLUMNS),
                ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ]
        )
        classifier = SKRandomForestClassifier(random_state=random_state)
        pipeline = SKPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1].tolist()
        metrics = evaluate_model(y_val.tolist(), val_probabilities)

        artifact = {
            "pipeline": pipeline,
            "feature_columns": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
            "target_column": TARGET_COLUMN,
            "trained_at": dt.datetime.utcnow().isoformat(),
            "training_params": {
                "test_size": test_size,
                "random_state": random_state,
                "algorithm": "sklearn.RandomForestClassifier",
                "n_estimators": classifier.n_estimators,
                "max_depth": classifier.max_depth,
            },
        }
        joblib.dump(artifact, model_path)
        write_base64_file(model_path, model_base64_path)

        class_balance = {
            "train": dict(Counter(int(v) for v in y_train)),
            "validation": dict(Counter(int(v) for v in y_val)),
        }
    else:
        train_records, val_records = stratified_split(
            records, test_size=test_size, random_state=random_state
        )

        preprocessor = Preprocessor()
        preprocessor.fit(train_records)
        X_train = preprocessor.transform(train_records)
        y_train = [rec[TARGET_COLUMN] for rec in train_records]
        X_val = preprocessor.transform(val_records)
        y_val = [rec[TARGET_COLUMN] for rec in val_records]

        model = SimpleRandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)

        val_probabilities = [prob for _, prob in model.predict_proba(X_val)]
        metrics = evaluate_model(y_val, val_probabilities)

        artifact = {
            "preprocessor": preprocessor,
            "model": model,
            "feature_names": preprocessor.feature_names,
            "target_column": TARGET_COLUMN,
            "trained_at": dt.datetime.utcnow().isoformat(),
            "training_params": {
                "test_size": test_size,
                "random_state": random_state,
                "algorithm": "SimpleRandomForestClassifier",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
            },
        }
        joblib.dump(artifact, model_path)
        write_base64_file(model_path, model_base64_path)

        class_balance = {
            "train": dict(Counter(y_train)),
            "validation": dict(Counter(y_val)),
        }

    metrics_payload = {
        "metrics": metrics,
        "n_train": len(y_train),
        "n_validation": len(y_val),
        "class_balance": class_balance,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lead conversion model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("backend/ml/data/lead_export.csv"),
        help="Path to the exported lead dataset in CSV format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/ml/models"),
        help="Directory where the trained model and metrics will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction used for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train(
        dataset_path=args.dataset,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(
        "Training complete. Model saved to {model_path}. Metrics saved to {metrics_path}.".format(
            **results
        )
    )
    print("Validation metrics:\n" + json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
