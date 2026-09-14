"""Small, preregistered statistics for activation-role probe qualification.

The native calibration protocol has twelve paired conversations per split.  A
pair contributes one mean unconditional reasoning probability for the native
reasoning segment and one for the native assistant/final segment.  This module
keeps the split, threshold, AUC, and bootstrap rules explicit so a test score
cannot silently tune the framing QA threshold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
from beartype import beartype

CALIBRATION_SPLIT: Final = "calibration"
TEST_SPLIT: Final = "test"
EXPECTED_CONVERSATIONS: Final = 12
BOOTSTRAP_SEED: Final = 0
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_QUANTILE_METHOD: Final = "linear"
MIN_ROLE_ACCURACY: Final = 0.75
MIN_DOCUMENT_MACRO_ACCURACY: Final = 0.75
MIN_THRESHOLD_REASONING_SENSITIVITY: Final = 0.75
MIN_THRESHOLD_FINAL_SPECIFICITY: Final = 0.75
MIN_TEST_AUC: Final = 0.85
MIN_AUC_BOOTSTRAP_LOWER: Final = 0.5


def _text(value: str, name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and trimmed")


def _rate(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")


@beartype
@dataclass(frozen=True, slots=True)
class PairedSegmentScores:
    """One unconditional ``P(reasoning)`` score for each segment of a dialogue."""

    split: str
    conversation_ids: tuple[str, ...]
    reasoning_scores: tuple[float, ...]
    final_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.split not in {CALIBRATION_SPLIT, TEST_SPLIT}:
            raise ValueError("split must be calibration or test")
        if not self.conversation_ids:
            raise ValueError("paired segment scores must contain conversations")
        if len(set(self.conversation_ids)) != len(self.conversation_ids):
            raise ValueError("conversation IDs must be unique within a split")
        for conversation_id in self.conversation_ids:
            _text(conversation_id, "conversation_id")
        if len(self.reasoning_scores) != len(self.conversation_ids) or len(
            self.final_scores
        ) != len(self.conversation_ids):
            raise ValueError("reasoning and final scores must align with conversation IDs")
        for name, values in (
            ("reasoning_scores", self.reasoning_scores),
            ("final_scores", self.final_scores),
        ):
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"{name} must contain finite probabilities between zero and one")

    @property
    def conversation_count(self) -> int:
        return len(self.conversation_ids)

    @property
    def fingerprint(self) -> str:
        """Return an order-independent identity for this exact score table."""
        rows = sorted(
            zip(
                self.conversation_ids,
                self.reasoning_scores,
                self.final_scores,
                strict=True,
            )
        )
        payload = {
            "split": self.split,
            "rows": [
                {"conversation_id": identifier, "reasoning": reasoning, "final": final}
                for identifier, reasoning, final in rows
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


@beartype
@dataclass(frozen=True, slots=True)
class PairedBootstrapAuc:
    """AUC and its fixed paired-conversation percentile bootstrap interval."""

    auc: float
    lower_95: float
    upper_95: float
    conversation_count: int
    seed: int = BOOTSTRAP_SEED
    replicates: int = BOOTSTRAP_REPLICATES
    quantile_method: str = BOOTSTRAP_QUANTILE_METHOD

    def __post_init__(self) -> None:
        for name, value in (
            ("auc", self.auc),
            ("lower_95", self.lower_95),
            ("upper_95", self.upper_95),
        ):
            _rate(value, name)
        if self.lower_95 > self.upper_95:
            raise ValueError("bootstrap lower bound cannot exceed upper bound")
        if self.conversation_count <= 0:
            raise ValueError("bootstrap AUC must contain at least one conversation")
        if self.seed != BOOTSTRAP_SEED:
            raise ValueError("paired bootstrap seed is fixed at zero")
        if self.replicates != BOOTSTRAP_REPLICATES:
            raise ValueError("paired bootstrap replicate count is fixed at 10000")
        if self.quantile_method != BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("paired bootstrap quantile method is fixed at linear")


@beartype
@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    """A frozen, calibration-only threshold for the framing QA direction."""

    calibration_fingerprint: str
    threshold: float | None
    calibration_auc: float
    balanced_accuracy: float | None
    reasoning_sensitivity: float | None
    final_specificity: float | None
    candidate_count: int
    usable: bool

    def __post_init__(self) -> None:
        if len(self.calibration_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.calibration_fingerprint
        ):
            raise ValueError("calibration_fingerprint must be a lowercase SHA-256 digest")
        _rate(self.calibration_auc, "calibration_auc")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        for name, value in (
            ("balanced_accuracy", self.balanced_accuracy),
            ("reasoning_sensitivity", self.reasoning_sensitivity),
            ("final_specificity", self.final_specificity),
        ):
            if value is not None:
                _rate(value, name)
        if self.usable:
            if self.threshold is None:
                raise ValueError("usable calibration must contain a threshold")
            if not np.isfinite(self.threshold):
                raise ValueError("usable threshold must be finite")
            if any(
                value is None
                for value in (
                    self.balanced_accuracy,
                    self.reasoning_sensitivity,
                    self.final_specificity,
                )
            ):
                raise ValueError("usable calibration must contain operating-point metrics")
        elif self.threshold is not None:
            raise ValueError("unusable calibration cannot contain a threshold")


@beartype
@dataclass(frozen=True, slots=True)
class NativeQualification:
    """The frozen native test gates, with no test-set threshold tuning."""

    scores: PairedSegmentScores
    minimum_role_accuracy: float
    document_macro_accuracy: float
    bootstrap_auc: PairedBootstrapAuc

    def __post_init__(self) -> None:
        if self.scores.split != TEST_SPLIT:
            raise ValueError("native qualification requires the untouched test split")
        if self.scores.conversation_count != EXPECTED_CONVERSATIONS:
            raise ValueError(
                f"native qualification requires {EXPECTED_CONVERSATIONS} conversations"
            )
        _rate(self.minimum_role_accuracy, "minimum_role_accuracy")
        _rate(self.document_macro_accuracy, "document_macro_accuracy")
        if self.bootstrap_auc.conversation_count != self.scores.conversation_count:
            raise ValueError("bootstrap AUC and native scores must contain the same conversations")
        if self.bootstrap_auc.auc != paired_segment_auc(self.scores):
            raise ValueError("bootstrap AUC does not match the native scores")

    @property
    def segment_passed(self) -> bool:
        """Return whether the untouched segment scores pass the rank gates."""
        return bool(
            self.bootstrap_auc.auc >= MIN_TEST_AUC
            and self.bootstrap_auc.lower_95 > MIN_AUC_BOOTSTRAP_LOWER
        )

    @property
    def passed(self) -> bool:
        """Return the legacy token-and-segment qualification result."""
        return bool(
            self.minimum_role_accuracy >= MIN_ROLE_ACCURACY
            and self.document_macro_accuracy >= MIN_DOCUMENT_MACRO_ACCURACY
            and self.segment_passed
        )


def _auc(reasoning: np.ndarray, final: np.ndarray) -> float:
    comparisons = (reasoning[:, None] > final[None, :]).astype(np.float64) + 0.5 * (
        reasoning[:, None] == final[None, :]
    )
    return float(comparisons.mean())


@beartype
def paired_segment_auc(scores: PairedSegmentScores) -> float:
    """Compute pooled segment AUC with 0.5 credit for exact score ties."""
    return _auc(
        np.asarray(scores.reasoning_scores, dtype=np.float64),
        np.asarray(scores.final_scores, dtype=np.float64),
    )


@beartype
def within_conversation_concordance(scores: PairedSegmentScores) -> float:
    """Report the paired direction rate, with half credit for ties.

    This is a descriptive within-conversation diagnostic.  It is deliberately
    separate from the preregistered pooled cross-conversation AUC and never
    contributes to native qualification.
    """
    reasoning = np.asarray(scores.reasoning_scores, dtype=np.float64)
    final = np.asarray(scores.final_scores, dtype=np.float64)
    comparisons = (reasoning > final).astype(np.float64) + 0.5 * (reasoning == final)
    return float(comparisons.mean())


@beartype
def paired_bootstrap_auc(scores: PairedSegmentScores) -> PairedBootstrapAuc:
    """Bootstrap pooled AUC by resampling whole conversations in fixed pairs."""
    conversation_count = scores.conversation_count
    rows = sorted(
        zip(
            scores.conversation_ids,
            scores.reasoning_scores,
            scores.final_scores,
            strict=True,
        )
    )
    reasoning = np.asarray([row[1] for row in rows], dtype=np.float64)
    final = np.asarray([row[2] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        conversation_count,
        size=(BOOTSTRAP_REPLICATES, conversation_count),
    )
    sampled_reasoning = reasoning[indices]
    sampled_final = final[indices]
    comparisons = (sampled_reasoning[:, :, None] > sampled_final[:, None, :]).astype(
        np.float64
    ) + 0.5 * (sampled_reasoning[:, :, None] == sampled_final[:, None, :])
    estimates = comparisons.mean(axis=(1, 2))
    bounds = np.quantile(
        estimates,
        (0.025, 0.975),
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return PairedBootstrapAuc(
        auc=paired_segment_auc(scores),
        lower_95=float(bounds[0]),
        upper_95=float(bounds[1]),
        conversation_count=conversation_count,
    )


def _threshold_candidates(scores: PairedSegmentScores) -> np.ndarray:
    values = np.unique(np.asarray(scores.reasoning_scores + scores.final_scores, dtype=np.float64))
    candidates = [float(np.nextafter(values[0], -np.inf))]
    candidates.extend(float(value) for value in values)
    candidates.extend(
        float((left + right) / 2) for left, right in zip(values[:-1], values[1:], strict=True)
    )
    candidates.append(float(np.nextafter(values[-1], np.inf)))
    return np.asarray(sorted(set(candidates)), dtype=np.float64)


@beartype
def calibrate_reasoning_threshold(scores: PairedSegmentScores) -> ThresholdCalibration:
    """Select a threshold from calibration scores only using a fixed tie rule."""
    if scores.split != CALIBRATION_SPLIT:
        raise ValueError("threshold calibration requires the calibration split")
    calibration_auc = paired_segment_auc(scores)
    best: tuple[tuple[float, float, float], float, float, float] | None = None
    candidates = _threshold_candidates(scores)
    reasoning = np.asarray(scores.reasoning_scores, dtype=np.float64)
    final = np.asarray(scores.final_scores, dtype=np.float64)
    for threshold in candidates:
        sensitivity = float(np.mean(reasoning >= threshold))
        specificity = float(np.mean(final < threshold))
        predicted = np.concatenate((reasoning, final)) >= threshold
        if predicted.all() or not predicted.any():
            continue
        balanced = (sensitivity + specificity) / 2
        key = (balanced, min(sensitivity, specificity), threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), sensitivity, specificity)
    if best is None or calibration_auc <= 0.5:
        return ThresholdCalibration(
            calibration_fingerprint=scores.fingerprint,
            threshold=None,
            calibration_auc=calibration_auc,
            balanced_accuracy=None,
            reasoning_sensitivity=None,
            final_specificity=None,
            candidate_count=len(candidates),
            usable=False,
        )
    _, threshold, sensitivity, specificity = best
    return ThresholdCalibration(
        calibration_fingerprint=scores.fingerprint,
        threshold=threshold,
        calibration_auc=calibration_auc,
        balanced_accuracy=(sensitivity + specificity) / 2,
        reasoning_sensitivity=sensitivity,
        final_specificity=specificity,
        candidate_count=len(candidates),
        usable=True,
    )


@beartype
def classify_reasoning(score: float, calibration: ThresholdCalibration) -> bool:
    """Apply the inclusive reasoning side of the frozen threshold."""
    if not calibration.usable or calibration.threshold is None:
        raise ValueError("calibration has no usable threshold")
    _rate(score, "score")
    return score >= calibration.threshold


@beartype
def qualify_native_test(
    scores: PairedSegmentScores,
    *,
    minimum_role_accuracy: float,
    document_macro_accuracy: float,
) -> NativeQualification:
    """Compute the fixed native-test AUC and role/document gates."""
    return NativeQualification(
        scores=scores,
        minimum_role_accuracy=minimum_role_accuracy,
        document_macro_accuracy=document_macro_accuracy,
        bootstrap_auc=paired_bootstrap_auc(scores),
    )
