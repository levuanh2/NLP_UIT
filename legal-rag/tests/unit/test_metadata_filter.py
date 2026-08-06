"""Metadata filtering test skeletons."""

import pytest


@pytest.mark.skip(reason="TODO(phase-implementation): implement confidence filtering")
def test_metadata_filter_applies_when_confident() -> None:
    # Arrange / Act / Assert
    pytest.fail("Enable after confidence-gated filtering is implemented.")


@pytest.mark.skip(reason="TODO(phase-implementation): implement confidence filtering")
def test_metadata_filter_skips_when_low_confidence() -> None:
    # Arrange / Act / Assert
    pytest.fail("Enable after low-confidence behavior is implemented.")


@pytest.mark.skip(reason="TODO(phase-implementation): implement empty filter fallback")
def test_metadata_filter_falls_back_to_full_corpus() -> None:
    # Arrange / Act / Assert
    pytest.fail("Enable after full-corpus fallback is implemented.")
