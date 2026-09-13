"""Frozen native-probe AUC, bootstrap, and threshold contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from reasonese.probe_statistics import (
    BOOTSTRAP_QUANTILE_METHOD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_SPLIT,
    EXPECTED_CONVERSATIONS,
    MIN_AUC_BOOTSTRAP_LOWER,
    MIN_DOCUMENT_MACRO_ACCURACY,
    MIN_ROLE_ACCURACY,
    MIN_TEST_AUC,
    MIN_THRESHOLD_FINAL_SPECIFICITY,
    MIN_THRESHOLD_REASONING_SENSITIVITY,
    TEST_SPLIT,
    NativeQualification,
    PairedBootstrapAuc,
    PairedSegmentScores,
    calibrate_reasoning_threshold,
    classify_reasoning,
    paired_bootstrap_auc,
    paired_segment_auc,
    qualify_native_test,
    within_conversation_concordance,
)


def _scores(
    split: str = CALIBRATION_SPLIT,
    reasoning: tuple[float, ...] | None = None,
    final: tuple[float, ...] | None = None,
) -> PairedSegmentScores:
    if reasoning is None:
        reasoning = tuple(0.8 + index / 100 for index in range(EXPECTED_CONVERSATIONS))
    if final is None:
        final = tuple(0.1 + index / 100 for index in range(EXPECTED_CONVERSATIONS))
    return PairedSegmentScores(
        split=split,
        conversation_ids=tuple(f"conversation-{index:02d}" for index in range(len(reasoning))),
        reasoning_scores=reasoning,
        final_scores=final,
    )


def test_scores_validate_pair_identity_and_fingerprint() -> None:
    scores = _scores()
    assert scores.conversation_count == EXPECTED_CONVERSATIONS
    assert len(scores.fingerprint) == 64
    reordered = replace(
        scores,
        conversation_ids=tuple(reversed(scores.conversation_ids)),
        reasoning_scores=tuple(reversed(scores.reasoning_scores)),
        final_scores=tuple(reversed(scores.final_scores)),
    )
    assert reordered.fingerprint == scores.fingerprint
    with pytest.raises(ValueError, match="unique"):
        replace(scores, conversation_ids=("duplicate",) * EXPECTED_CONVERSATIONS)
    with pytest.raises(ValueError, match="align"):
        replace(scores, final_scores=scores.final_scores[:-1])
    with pytest.raises(ValueError, match="probabilities"):
        replace(scores, reasoning_scores=(0.9,) * (EXPECTED_CONVERSATIONS - 1) + (1.1,))
    with pytest.raises(ValueError, match="split"):
        replace(scores, split="test-set")


def test_auc_uses_unconditional_scores_and_half_credit_for_ties() -> None:
    scores = _scores(
        reasoning=(0.8,) * EXPECTED_CONVERSATIONS,
        final=(0.2,) * EXPECTED_CONVERSATIONS,
    )
    assert paired_segment_auc(scores) == 1.0
    ties = replace(
        scores,
        reasoning_scores=(0.5,) * EXPECTED_CONVERSATIONS,
        final_scores=(0.5,) * EXPECTED_CONVERSATIONS,
    )
    assert paired_segment_auc(ties) == 0.5
    partial = replace(
        scores,
        reasoning_scores=(0.2,) * 6 + (0.8,) * 6,
        final_scores=(0.1,) * 9 + (0.9,) * 3,
    )
    assert paired_segment_auc(partial) == pytest.approx(0.75)


def test_bootstrap_is_paired_deterministic_and_uses_linear_quantiles() -> None:
    scores = _scores(
        reasoning=tuple(0.51 + index / 100 for index in range(EXPECTED_CONVERSATIONS)),
        final=tuple(0.49 - index / 100 for index in range(EXPECTED_CONVERSATIONS)),
    )
    first = paired_bootstrap_auc(scores)
    second = paired_bootstrap_auc(scores)
    assert first == second
    assert first.seed == BOOTSTRAP_SEED
    assert first.replicates == BOOTSTRAP_REPLICATES
    assert first.quantile_method == BOOTSTRAP_QUANTILE_METHOD
    assert first.auc == 1.0
    assert first.lower_95 == 1.0
    assert first.upper_95 == 1.0


def test_bootstrap_is_invariant_to_input_row_permutation() -> None:
    scores = _scores(
        reasoning=tuple(0.51 + index / 100 for index in range(EXPECTED_CONVERSATIONS)),
        final=tuple(0.49 - index / 100 for index in range(EXPECTED_CONVERSATIONS)),
    )
    reordered = replace(
        scores,
        conversation_ids=tuple(reversed(scores.conversation_ids)),
        reasoning_scores=tuple(reversed(scores.reasoning_scores)),
        final_scores=tuple(reversed(scores.final_scores)),
    )
    assert paired_bootstrap_auc(reordered) == paired_bootstrap_auc(scores)


def test_bootstrap_preserves_pairs_for_cross_conversation_auc() -> None:
    # Each conversation is correctly ordered, but pooling all segments gives
    # AUC 0.75. A within-conversation statistic would incorrectly report 1.0.
    scores = _scores(
        reasoning=(0.9,) * 6 + (0.1,) * 6,
        final=(0.8,) * 6 + (0.0,) * 6,
    )
    assert paired_segment_auc(scores) == pytest.approx(0.75)
    result = paired_bootstrap_auc(scores)
    assert result.auc == pytest.approx(0.75)
    assert 0.5 <= result.lower_95 <= result.auc <= result.upper_95 <= 1.0


def test_within_conversation_concordance_is_report_only_and_credits_ties() -> None:
    scores = _scores(
        reasoning=(0.9, 0.8, 0.5, 0.1) + (0.5,) * 8,
        final=(0.1, 0.8, 0.5, 0.9) + (0.5,) * 8,
    )
    # Four first pairs contribute 1, .5, .5, 0; eight ties contribute .5.
    assert within_conversation_concordance(scores) == pytest.approx(0.5)


def test_calibration_freezes_threshold_with_explicit_tie_rule() -> None:
    scores = _scores(
        reasoning=(0.8,) * 6 + (0.7,) * 6,
        final=(0.2,) * 6 + (0.3,) * 6,
    )
    calibration = calibrate_reasoning_threshold(scores)
    assert calibration.usable
    assert calibration.calibration_fingerprint == scores.fingerprint
    threshold = calibration.threshold
    assert threshold is not None
    assert threshold == pytest.approx(0.7)
    assert classify_reasoning(threshold, calibration)
    assert classify_reasoning(threshold - 1e-9, calibration) is False
    assert classify_reasoning(threshold + 1e-9, calibration)
    test_scores = replace(
        scores,
        split=TEST_SPLIT,
        reasoning_scores=tuple(0.01 for _ in scores.reasoning_scores),
        final_scores=tuple(0.99 for _ in scores.final_scores),
    )
    assert calibrate_reasoning_threshold(scores).threshold == calibration.threshold
    assert test_scores.fingerprint != calibration.calibration_fingerprint


def test_calibration_tie_breaks_to_highest_threshold() -> None:
    # Every candidate has balanced accuracy 0.5; the largest finite candidate
    # that still has both predicted classes is selected deterministically.
    reasoning = (0.1,) * 9 + (0.2,) * 3
    final = (0.1,) * 12
    calibration = calibrate_reasoning_threshold(_scores(reasoning=reasoning, final=final))
    assert calibration.usable
    assert calibration.threshold == pytest.approx(0.2)
    assert calibration.reasoning_sensitivity == pytest.approx(0.25)
    assert calibration.final_specificity == pytest.approx(1.0)


def test_degenerate_calibration_disables_enforced_qa() -> None:
    scores = _scores(
        reasoning=(0.5,) * EXPECTED_CONVERSATIONS,
        final=(0.5,) * EXPECTED_CONVERSATIONS,
    )
    calibration = calibrate_reasoning_threshold(scores)
    assert calibration.calibration_auc == 0.5
    assert calibration.usable is False
    assert calibration.threshold is None
    with pytest.raises(ValueError, match="usable threshold"):
        classify_reasoning(0.5, calibration)


def test_primitives_support_small_synthetic_pairs_but_native_gate_requires_twelve() -> None:
    with pytest.raises(ValueError, match="calibration split"):
        calibrate_reasoning_threshold(_scores(TEST_SPLIT))
    small = _scores(reasoning=(0.8, 0.9), final=(0.2, 0.1))
    assert paired_segment_auc(small) == 1.0
    assert paired_bootstrap_auc(small).conversation_count == 2
    assert calibrate_reasoning_threshold(small).usable
    with pytest.raises(ValueError, match="12"):
        qualify_native_test(
            replace(small, split=TEST_SPLIT),
            calibration=calibrate_reasoning_threshold(small),
            minimum_role_accuracy=0.9,
            document_macro_accuracy=0.9,
        )


def test_native_test_qualification_uses_only_frozen_gates() -> None:
    scores = _scores(
        split=TEST_SPLIT,
        reasoning=(0.9,) * EXPECTED_CONVERSATIONS,
        final=(0.1,) * EXPECTED_CONVERSATIONS,
    )
    qualification = qualify_native_test(
        scores,
        calibration=calibrate_reasoning_threshold(replace(scores, split=CALIBRATION_SPLIT)),
        minimum_role_accuracy=MIN_ROLE_ACCURACY,
        document_macro_accuracy=MIN_DOCUMENT_MACRO_ACCURACY,
    )
    assert qualification.passed
    assert qualification.threshold_reasoning_sensitivity == 1.0
    assert qualification.threshold_final_specificity == 1.0
    assert qualification.bootstrap_auc.auc >= MIN_TEST_AUC
    assert qualification.bootstrap_auc.lower_95 > MIN_AUC_BOOTSTRAP_LOWER

    exact_role_failure = replace(qualification, minimum_role_accuracy=MIN_ROLE_ACCURACY - 1e-12)
    assert exact_role_failure.passed is False
    exact_doc_failure = replace(
        qualification,
        document_macro_accuracy=MIN_DOCUMENT_MACRO_ACCURACY - 1e-12,
    )
    assert exact_doc_failure.passed is False
    assert (
        replace(
            qualification,
            threshold_reasoning_sensitivity=np.nextafter(
                MIN_THRESHOLD_REASONING_SENSITIVITY, -np.inf
            ),
        ).passed
        is False
    )
    assert (
        replace(
            qualification,
            threshold_final_specificity=np.nextafter(MIN_THRESHOLD_FINAL_SPECIFICITY, -np.inf),
        ).passed
        is False
    )
    with pytest.raises(ValueError, match="test split"):
        qualify_native_test(
            replace(scores, split=CALIBRATION_SPLIT),
            calibration=calibrate_reasoning_threshold(replace(scores, split=CALIBRATION_SPLIT)),
            minimum_role_accuracy=0.9,
            document_macro_accuracy=0.9,
        )


def test_native_gate_rejects_a_ranked_probe_whose_frozen_threshold_does_not_transfer() -> None:
    calibration_scores = _scores(
        reasoning=(0.99,) * EXPECTED_CONVERSATIONS,
        final=(0.01,) * EXPECTED_CONVERSATIONS,
    )
    calibration = calibrate_reasoning_threshold(calibration_scores)
    assert calibration.threshold == 0.99
    test_scores = _scores(
        split=TEST_SPLIT,
        reasoning=(0.95,) * EXPECTED_CONVERSATIONS,
        final=(0.01,) * EXPECTED_CONVERSATIONS,
    )

    qualification = qualify_native_test(
        test_scores,
        calibration=calibration,
        minimum_role_accuracy=1.0,
        document_macro_accuracy=1.0,
    )

    assert qualification.bootstrap_auc.auc == 1.0
    assert qualification.threshold_reasoning_sensitivity == 0.0
    assert qualification.threshold_final_specificity == 1.0
    assert qualification.passed is False


def test_native_qualification_boundary_comparisons_are_explicit() -> None:
    scores = _scores(split=TEST_SPLIT)
    auc = PairedBootstrapAuc(
        auc=paired_segment_auc(scores),
        lower_95=np.nextafter(MIN_AUC_BOOTSTRAP_LOWER, np.inf),
        upper_95=0.9,
        conversation_count=EXPECTED_CONVERSATIONS,
    )
    assert NativeQualification(scores, 0.75, 0.75, 0.75, 0.75, auc).passed
    lower_equal = replace(auc, lower_95=MIN_AUC_BOOTSTRAP_LOWER)
    assert NativeQualification(scores, 0.75, 0.75, 0.75, 0.75, lower_equal).passed is False


def test_native_qualification_rejects_a_mismatched_bootstrap_point_auc() -> None:
    scores = _scores(split=TEST_SPLIT)
    mismatched = PairedBootstrapAuc(
        auc=MIN_TEST_AUC,
        lower_95=0.6,
        upper_95=0.9,
        conversation_count=EXPECTED_CONVERSATIONS,
    )
    with pytest.raises(ValueError, match="does not match"):
        NativeQualification(scores, 0.9, 0.9, 0.9, 0.9, mismatched)


def test_bootstrap_configuration_cannot_be_changed() -> None:
    with pytest.raises(ValueError, match="seed"):
        PairedBootstrapAuc(1.0, 1.0, 1.0, EXPECTED_CONVERSATIONS, seed=1)
    with pytest.raises(ValueError, match="replicate"):
        PairedBootstrapAuc(1.0, 1.0, 1.0, EXPECTED_CONVERSATIONS, replicates=100)
    with pytest.raises(ValueError, match="quantile"):
        PairedBootstrapAuc(1.0, 1.0, 1.0, EXPECTED_CONVERSATIONS, quantile_method="nearest")
