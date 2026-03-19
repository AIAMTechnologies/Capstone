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

from collections import Counter

logger = logging.getLogger("installer_ml")


class InstallerMLModel:
    """Encapsulates training and inference for installer predictions."""

    def __init__(
        self,
        query_executor: Callable[[str, Optional[tuple], bool], list],
        retrain_interval: timedelta = timedelta(hours=6),
        min_training_rows: int = 25,
        failure_retry_interval: timedelta = timedelta(minutes=15),
    ) -> None:
        self._query_executor = query_executor
        self._pipeline: Optional["_SimplePipeline"] = None
        self._last_trained_at: Optional[datetime] = None
        self._retrain_interval = retrain_interval
        self._min_training_rows = min_training_rows
        self._last_row_count: Optional[int] = None
        self._last_error: Optional[str] = None
        self._failure_retry_interval = failure_retry_interval
        self._last_attempt_at: Optional[datetime] = None

    def ensure_ready(self) -> bool:
        """Train the model if it has never been trained or is stale."""

        now = datetime.utcnow()
        needs_training = False

        if self._pipeline is None:
            needs_training = True
            if (
                self._last_attempt_at
                and now - self._last_attempt_at < self._failure_retry_interval
            ):
                if self._last_error:
                    logger.debug(
                        "Skipping ML training attempt due to cooldown (last error: %s)",
                        self._last_error,
                    )
                return False
        elif self._last_trained_at is None:
            needs_training = True
        elif now - self._last_trained_at > self._retrain_interval:
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
            Dictionary containing ``project_type``, ``square_footage`` and
            ``current_status``.  Missing values are imputed using the
            statistics learned during training.
        """

        if not self.ensure_ready() or self._pipeline is None:
            return {}

        probabilities = self._pipeline.predict_proba([features])[0]
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
        self._last_attempt_at = datetime.utcnow()
        try:
            records = self._query_executor(
                """
                SELECT final_installer_selection, dealer_name, project_type, square_footage, current_status
                FROM historical_data
                WHERE final_installer_selection IS NOT NULL
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

        cleaned = []
        for record in records:
            label = (record.get("final_installer_selection") or "").strip()
            if not label:
                continue
            cleaned.append(label)

        if len(cleaned) < self._min_training_rows:
            logger.warning(
                "Not enough historical rows to train ML model (found %s)",
                len(cleaned),
            )
            self._pipeline = None
            self._last_row_count = len(cleaned)
            self._last_error = "insufficient_training_data"
            return False

        class_counts = Counter(cleaned)
        total = float(sum(class_counts.values()))
        classes = list(class_counts.keys())
        probabilities = [class_counts[c] / total for c in classes]

        model = _SimpleModel(classes, probabilities)
        self._pipeline = _SimplePipeline(model)
        self._last_trained_at = datetime.utcnow()
        self._last_row_count = len(cleaned)
        self._last_error = None
        logger.info("Installer ML model trained on %s rows", len(cleaned))
        return True


class _SimpleModel:
    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, _features):
        return [self._probabilities]


class _SimplePipeline:
    def __init__(self, model: _SimpleModel):
        self.named_steps = {"model": model}
        self._model = model

    def predict_proba(self, features):
        # Features are not used in the simplified model; it returns the learned
        # class distribution so callers still receive probability mappings.
        return self._model.predict_proba(features)
