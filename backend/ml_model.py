"""Utility module that encapsulates the machine learning model used to
recommend installers for incoming leads.

The model is trained on historical data stored in the ``historical_data``
table.  It learns relationships between project attributes and the dealer
that ultimately completed the job.  The predictor is kept deliberately
light-weight so it can be refreshed in memory when the API boots or when new
data becomes available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger("installer_ml")


class InstallerMLModel:
    """Encapsulates training and inference for installer predictions."""

    def __init__(
        self,
        query_executor: Callable[[str, Optional[tuple], bool], list],
        retrain_interval: timedelta = timedelta(hours=6),
        min_training_rows: int = 25,
    ) -> None:
        self._query_executor = query_executor
        self._pipeline: Optional[Pipeline] = None
        self._last_trained_at: Optional[datetime] = None
        self._retrain_interval = retrain_interval
        self._min_training_rows = min_training_rows
        self._last_row_count: Optional[int] = None
        self._last_error: Optional[str] = None

    def ensure_ready(self) -> bool:
        """Train the model if it has never been trained or is stale."""

        needs_training = self._pipeline is None
        if self._last_trained_at is None:
            needs_training = True
        elif datetime.utcnow() - self._last_trained_at > self._retrain_interval:
            needs_training = True

        if needs_training:
            try:
                return self._train_model()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to train installer ML model: %s", exc)
                self._pipeline = None
                self._last_error = str(exc)
                return False
        return True

    def train(self, force: bool = True) -> bool:
        """Trigger training manually.

        Parameters
        ----------
        force:
            When True the model will be retrained immediately even if the
            cached pipeline is considered fresh.  When False it behaves like
            :py:meth:`ensure_ready`.
        """

        if force:
            return self._train_model()
        return self.ensure_ready()

    def status(self) -> Dict[str, Optional[str]]:
        """Return metadata describing the most recent training run."""

        return {
            "trained": self._pipeline is not None,
            "last_trained_at": self._last_trained_at,
            "training_rows": self._last_row_count,
            "last_error": self._last_error,
        }

    def predict_probabilities(self, features: Dict[str, Optional[float]]) -> Dict[str, float]:
        """Return probability for each dealer given project features.

        Parameters
        ----------
        features:
            Dictionary containing ``project_type``, ``product_type``,
            ``square_footage`` and ``current_status``.  Missing values are
            imputed using the statistics learned during training.
        """

        if not self.ensure_ready() or self._pipeline is None:
            return {}

        model_features = {
            "project_type": features.get("project_type") or "Unknown",
            "product_type": features.get("product_type") or "Unknown",
            "square_footage": features.get("square_footage"),
            "current_status": features.get("current_status") or "Unknown",
        }
        frame = pd.DataFrame([model_features])

        probabilities = self._pipeline.predict_proba(frame)[0]
        labels = list(self._pipeline.named_steps["model"].classes_)
        result: Dict[str, float] = {}
        for label, prob in zip(labels, probabilities):
            if label is None:
                continue
            normalized = str(label).strip()
            value = float(prob)
            result[normalized] = value
            result[normalized.lower()] = value
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_model(self) -> bool:
        """Fetch data from the database and train the estimator."""

        logger.info("Training installer ML model from historical data")
        try:
            records = self._query_executor(
                """
                SELECT dealer_name, project_type, product_type, square_footage, current_status
                FROM historical_data
                WHERE dealer_name IS NOT NULL
                """,
                None,
                True,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Unable to fetch historical data for ML training: %s", exc)
            self._pipeline = None
            return False

        if not records or len(records) < self._min_training_rows:
            logger.warning(
                "Not enough historical rows to train ML model (found %s)",
                len(records) if records else 0,
            )
            self._pipeline = None
            self._last_row_count = len(records) if records else 0
            self._last_error = "insufficient_training_data"
            return False

        frame = pd.DataFrame(records)
        frame["square_footage"] = pd.to_numeric(frame.get("square_footage"), errors="coerce")
        frame["project_type"] = frame.get("project_type", "Unknown").fillna("Unknown")
        frame["product_type"] = frame.get("product_type", "Unknown").fillna("Unknown")
        frame["current_status"] = frame.get("current_status", "Unknown").fillna("Unknown")
        frame = frame.dropna(subset=["dealer_name"])

        if frame.empty or frame["dealer_name"].nunique() < 2:
            logger.warning("Insufficient label diversity to train ML model")
            self._pipeline = None
            self._last_row_count = len(frame)
            self._last_error = "insufficient_label_diversity"
            return False

        features = frame[["project_type", "product_type", "square_footage", "current_status"]]
        labels = frame["dealer_name"].astype(str).str.strip()

        categorical_features = ["project_type", "product_type", "current_status"]
        numeric_features = ["square_footage"]

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, categorical_features),
                ("numeric", numeric_transformer, numeric_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        pipeline.fit(features, labels)
        self._pipeline = pipeline
        self._last_trained_at = datetime.utcnow()
        self._last_row_count = len(frame)
        self._last_error = None
        logger.info("Installer ML model trained on %s rows", len(frame))
        return True
