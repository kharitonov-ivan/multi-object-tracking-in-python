import pytest

from src.common.hypothesis_reduction import HypothesisReduction


def test_prune():
    # Mock data
    hypotheses_weights = [-10.0, -5.0, 0.0, 5.0, 10.0]
    multi_hypotheses = ["h1", "h2", "h3", "h4", "h5"]
    threshold = 0.0

    # Expected output
    expected_hypotheses_weights = [5.0, 10.0]
    expected_multi_hypotheses = ["h4", "h5"]

    # Call the prune function
    new_hypotheses_weights, new_multi_hypotheses = HypothesisReduction.prune(hypotheses_weights, multi_hypotheses, threshold)

    # Check if the output is as expected
    assert new_hypotheses_weights == expected_hypotheses_weights, f"Expected {expected_hypotheses_weights} but got {new_hypotheses_weights}"
    assert new_multi_hypotheses == expected_multi_hypotheses, f"Expected {expected_multi_hypotheses} but got {new_multi_hypotheses}"


def test_cap():
    # Mock data
    hypotheses_weights = [-10.0, -5.0, 0.0, 5.0, 10.0]
    multi_hypotheses = ["h1", "h2", "h3", "h4", "h5"]
    top_k = 3

    # Expected output
    expected_hypotheses_weights = [-10.0, -5.0, 0.0]
    expected_multi_hypotheses = ["h1", "h2", "h3"]

    # Call the cap function
    new_hypotheses_weights, new_multi_hypotheses = HypothesisReduction.cap(hypotheses_weights, multi_hypotheses, top_k)

    # Check if the output is as expected
    assert new_hypotheses_weights == expected_hypotheses_weights, f"Expected {expected_hypotheses_weights} but got {new_hypotheses_weights}"
    assert new_multi_hypotheses == expected_multi_hypotheses, f"Expected {expected_multi_hypotheses} but got {new_multi_hypotheses}"


def test_cap_negative():
    # Negative case
    with pytest.raises(AssertionError):
        HypothesisReduction.cap([], [], -1)
