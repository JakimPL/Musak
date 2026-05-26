import pytest

from musak_model.conditioning.structural.config import DurationDenominatorBucketConfig, IntegerBucketConfig


def test_structural_control_config_rejects_unsorted_thresholds() -> None:
    with pytest.raises(ValueError, match="sorted"):
        IntegerBucketConfig(thresholds=(2, 1))


def test_duration_denominator_bucket_config_rejects_non_power_of_two_thresholds() -> None:
    with pytest.raises(ValueError, match="powers of two"):
        DurationDenominatorBucketConfig(thresholds=(16, 12, 8))


def test_duration_denominator_bucket_config_rejects_string_thresholds() -> None:
    with pytest.raises(ValueError, match="integer denominators"):
        DurationDenominatorBucketConfig.model_validate({"thresholds": ["1/16", "1/8"]})


def test_duration_denominator_bucket_config_rejects_duration_order_mismatch() -> None:
    with pytest.raises(ValueError, match="shortest to longest duration"):
        DurationDenominatorBucketConfig(thresholds=(4, 8, 16))
