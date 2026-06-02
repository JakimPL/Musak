from numpy.random import default_rng

from musak_model.synthetic.processes.density import RhythmicDensityConfig, RhythmicDensitySampler


def _sampler(*, amplitude: float = 1.0, basis_count: int = 2, decay: float = 1.0) -> RhythmicDensitySampler:
    return RhythmicDensitySampler(
        config=RhythmicDensityConfig(amplitude=amplitude, basis_count=basis_count, decay=decay)
    )


def test_sample_returns_one_offset_per_position() -> None:
    offsets = _sampler().sample(length=8, rng=default_rng(0))

    assert len(offsets) == 8
    assert all(isinstance(offset, float) for offset in offsets)


def test_zero_amplitude_yields_no_drift() -> None:
    offsets = _sampler(amplitude=0.0).sample(length=8, rng=default_rng(0))

    assert offsets == (0.0,) * 8


def test_sample_is_deterministic_for_a_seed() -> None:
    first = _sampler().sample(length=12, rng=default_rng(5))
    second = _sampler().sample(length=12, rng=default_rng(5))

    assert first == second


def test_empty_length_returns_empty() -> None:
    assert _sampler().sample(length=0, rng=default_rng(0)) == ()


def test_low_basis_count_drifts_gently() -> None:
    offsets = _sampler(amplitude=2.0, basis_count=1).sample(length=64, rng=default_rng(1))

    sign_changes = sum(
        1 for current, following in zip(offsets, offsets[1:], strict=False) if (current > 0) != (following > 0)
    )
    assert sign_changes <= 1
