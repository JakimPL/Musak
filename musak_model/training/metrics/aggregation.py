from __future__ import annotations

from dataclasses import dataclass
from math import exp

from musak_model.conditioning.harmony.relations import HARMONIC_RELATION_CLASS_COUNT
from musak_model.training.metrics.schema import BatchMetrics, EpochSplitMetrics


@dataclass
class MetricsAccumulator:
    loss_sum: float = 0.0
    token_count: int = 0
    exact_match_count: int = 0
    token_kind_match_count: int | None = None
    event_kind_loss_sum: float | None = None
    event_kind_loss_target_count: int | None = None
    duration_loss_sum: float | None = None
    duration_loss_target_count: int | None = None
    degree_loss_sum: float | None = None
    degree_loss_target_count: int | None = None
    accidental_loss_sum: float | None = None
    accidental_loss_target_count: int | None = None
    octave_offset_loss_sum: float | None = None
    octave_offset_loss_target_count: int | None = None
    hand_loss_sum: float | None = None
    hand_loss_target_count: int | None = None
    duration_match_count: int | None = None
    duration_target_count: int | None = None
    degree_match_count: int | None = None
    degree_target_count: int | None = None
    accidental_match_count: int | None = None
    accidental_target_count: int | None = None
    octave_offset_match_count: int | None = None
    octave_offset_target_count: int | None = None
    hand_match_count: int | None = None
    hand_target_count: int | None = None
    musical_auxiliary_loss_sum: float | None = None
    musical_auxiliary_target_count: int | None = None
    note_density_loss_sum: float | None = None
    note_density_match_count: int | None = None
    note_density_target_count: int | None = None
    rhythmic_diversity_loss_sum: float | None = None
    rhythmic_diversity_match_count: int | None = None
    rhythmic_diversity_target_count: int | None = None
    voice_independence_loss_sum: float | None = None
    voice_independence_match_count: int | None = None
    voice_independence_target_count: int | None = None
    uses_accidentals_loss_sum: float | None = None
    uses_accidentals_match_count: int | None = None
    uses_accidentals_target_count: int | None = None
    dotted_duration_loss_sum: float | None = None
    dotted_duration_match_count: int | None = None
    dotted_duration_target_count: int | None = None
    hand_span_loss_sum: float | None = None
    hand_span_match_count: int | None = None
    hand_span_target_count: int | None = None
    bar_note_density_loss_sum: float | None = None
    bar_note_density_match_count: int | None = None
    bar_note_density_target_count: int | None = None
    bar_rhythmic_diversity_loss_sum: float | None = None
    bar_rhythmic_diversity_match_count: int | None = None
    bar_rhythmic_diversity_target_count: int | None = None
    bar_voice_independence_loss_sum: float | None = None
    bar_voice_independence_match_count: int | None = None
    bar_voice_independence_target_count: int | None = None
    bar_uses_accidentals_loss_sum: float | None = None
    bar_uses_accidentals_match_count: int | None = None
    bar_uses_accidentals_target_count: int | None = None
    bar_dotted_duration_loss_sum: float | None = None
    bar_dotted_duration_match_count: int | None = None
    bar_dotted_duration_target_count: int | None = None
    bar_hand_span_loss_sum: float | None = None
    bar_hand_span_match_count: int | None = None
    bar_hand_span_target_count: int | None = None
    harmonic_relation_loss_sum: float | None = None
    harmonic_relation_match_count: int | None = None
    harmonic_relation_target_count: int | None = None
    harmonic_relation_macro_f1_sum: float | None = None
    harmonic_relation_target_counts: list[int] | None = None
    harmonic_relation_prediction_counts: list[int] | None = None
    harmonic_plan_reconstruction_loss_sum: float | None = None
    harmonic_plan_reconstruction_target_count: int | None = None
    harmonic_plan_reconstruction_harmonic_function_match_count: int | None = None
    harmonic_plan_reconstruction_harmonic_function_target_count: int | None = None
    harmonic_plan_reconstruction_root_degree_match_count: int | None = None
    harmonic_plan_reconstruction_root_degree_target_count: int | None = None
    harmonic_plan_reconstruction_quality_match_count: int | None = None
    harmonic_plan_reconstruction_quality_target_count: int | None = None
    harmonic_plan_reconstruction_extension_match_count: int | None = None
    harmonic_plan_reconstruction_extension_target_count: int | None = None
    harmonic_plan_reconstruction_cadence_strength_match_count: int | None = None
    harmonic_plan_reconstruction_cadence_strength_target_count: int | None = None
    harmonic_plan_contrastive_loss_sum: float | None = None
    harmonic_plan_contrastive_match_count: int | None = None
    harmonic_plan_contrastive_target_count: int | None = None
    harmonic_plan_contrastive_positive_similarity_sum: float | None = None
    harmonic_plan_contrastive_negative_similarity_sum: float | None = None
    harmony_gate_mean_sum: float | None = None
    harmony_gate_token_count: int | None = None
    validity_penalty_loss_sum: float | None = None
    invalid_probability_mass_sum: float | None = None
    invalid_target_count: int | None = None
    validity_penalty_token_count: int | None = None
    cnn_gradient_norm_sum: float | None = None
    gru_gradient_norm_sum: float | None = None
    transformer_gradient_norm_sum: float | None = None

    def add(self, batch_metrics: BatchMetrics) -> None:
        self.loss_sum += batch_metrics.loss * batch_metrics.token_count
        self.token_count += batch_metrics.token_count
        self.exact_match_count += batch_metrics.exact_match_count
        if batch_metrics.token_kind_match_count is not None:
            if self.token_kind_match_count is None:
                self.token_kind_match_count = 0

            self.token_kind_match_count += batch_metrics.token_kind_match_count

        self.event_kind_loss_sum, self.event_kind_loss_target_count = _add_optional_weighted_loss(
            self.event_kind_loss_sum,
            self.event_kind_loss_target_count,
            value=batch_metrics.event_kind_loss,
            target_count=batch_metrics.event_kind_loss_target_count,
        )
        self.duration_loss_sum, self.duration_loss_target_count = _add_optional_weighted_loss(
            self.duration_loss_sum,
            self.duration_loss_target_count,
            value=batch_metrics.duration_loss,
            target_count=batch_metrics.duration_loss_target_count,
        )
        self.degree_loss_sum, self.degree_loss_target_count = _add_optional_weighted_loss(
            self.degree_loss_sum,
            self.degree_loss_target_count,
            value=batch_metrics.degree_loss,
            target_count=batch_metrics.degree_loss_target_count,
        )
        self.accidental_loss_sum, self.accidental_loss_target_count = _add_optional_weighted_loss(
            self.accidental_loss_sum,
            self.accidental_loss_target_count,
            value=batch_metrics.accidental_loss,
            target_count=batch_metrics.accidental_loss_target_count,
        )
        self.octave_offset_loss_sum, self.octave_offset_loss_target_count = _add_optional_weighted_loss(
            self.octave_offset_loss_sum,
            self.octave_offset_loss_target_count,
            value=batch_metrics.octave_offset_loss,
            target_count=batch_metrics.octave_offset_loss_target_count,
        )
        self.hand_loss_sum, self.hand_loss_target_count = _add_optional_weighted_loss(
            self.hand_loss_sum,
            self.hand_loss_target_count,
            value=batch_metrics.hand_loss,
            target_count=batch_metrics.hand_loss_target_count,
        )
        self.duration_match_count, self.duration_target_count = _add_optional_count_pair(
            self.duration_match_count,
            self.duration_target_count,
            match_count=batch_metrics.duration_match_count,
            target_count=batch_metrics.duration_target_count,
        )
        self.degree_match_count, self.degree_target_count = _add_optional_count_pair(
            self.degree_match_count,
            self.degree_target_count,
            match_count=batch_metrics.degree_match_count,
            target_count=batch_metrics.degree_target_count,
        )
        self.accidental_match_count, self.accidental_target_count = _add_optional_count_pair(
            self.accidental_match_count,
            self.accidental_target_count,
            match_count=batch_metrics.accidental_match_count,
            target_count=batch_metrics.accidental_target_count,
        )
        self.octave_offset_match_count, self.octave_offset_target_count = _add_optional_count_pair(
            self.octave_offset_match_count,
            self.octave_offset_target_count,
            match_count=batch_metrics.octave_offset_match_count,
            target_count=batch_metrics.octave_offset_target_count,
        )
        self.hand_match_count, self.hand_target_count = _add_optional_count_pair(
            self.hand_match_count,
            self.hand_target_count,
            match_count=batch_metrics.hand_match_count,
            target_count=batch_metrics.hand_target_count,
        )
        self.musical_auxiliary_loss_sum, self.musical_auxiliary_target_count = _add_optional_weighted_loss(
            self.musical_auxiliary_loss_sum,
            self.musical_auxiliary_target_count,
            value=batch_metrics.musical_auxiliary_loss,
            target_count=batch_metrics.musical_auxiliary_target_count,
        )
        self.note_density_loss_sum, self.note_density_match_count, self.note_density_target_count = (
            _add_optional_auxiliary_metric(
                self.note_density_loss_sum,
                self.note_density_match_count,
                self.note_density_target_count,
                value=batch_metrics.note_density_loss,
                match_count=batch_metrics.note_density_match_count,
                target_count=batch_metrics.note_density_target_count,
            )
        )
        self.rhythmic_diversity_loss_sum, self.rhythmic_diversity_match_count, self.rhythmic_diversity_target_count = (
            _add_optional_auxiliary_metric(
                self.rhythmic_diversity_loss_sum,
                self.rhythmic_diversity_match_count,
                self.rhythmic_diversity_target_count,
                value=batch_metrics.rhythmic_diversity_loss,
                match_count=batch_metrics.rhythmic_diversity_match_count,
                target_count=batch_metrics.rhythmic_diversity_target_count,
            )
        )
        self.voice_independence_loss_sum, self.voice_independence_match_count, self.voice_independence_target_count = (
            _add_optional_auxiliary_metric(
                self.voice_independence_loss_sum,
                self.voice_independence_match_count,
                self.voice_independence_target_count,
                value=batch_metrics.voice_independence_loss,
                match_count=batch_metrics.voice_independence_match_count,
                target_count=batch_metrics.voice_independence_target_count,
            )
        )
        self.uses_accidentals_loss_sum, self.uses_accidentals_match_count, self.uses_accidentals_target_count = (
            _add_optional_auxiliary_metric(
                self.uses_accidentals_loss_sum,
                self.uses_accidentals_match_count,
                self.uses_accidentals_target_count,
                value=batch_metrics.uses_accidentals_loss,
                match_count=batch_metrics.uses_accidentals_match_count,
                target_count=batch_metrics.uses_accidentals_target_count,
            )
        )
        self.dotted_duration_loss_sum, self.dotted_duration_match_count, self.dotted_duration_target_count = (
            _add_optional_auxiliary_metric(
                self.dotted_duration_loss_sum,
                self.dotted_duration_match_count,
                self.dotted_duration_target_count,
                value=batch_metrics.dotted_duration_loss,
                match_count=batch_metrics.dotted_duration_match_count,
                target_count=batch_metrics.dotted_duration_target_count,
            )
        )
        self.hand_span_loss_sum, self.hand_span_match_count, self.hand_span_target_count = (
            _add_optional_auxiliary_metric(
                self.hand_span_loss_sum,
                self.hand_span_match_count,
                self.hand_span_target_count,
                value=batch_metrics.hand_span_loss,
                match_count=batch_metrics.hand_span_match_count,
                target_count=batch_metrics.hand_span_target_count,
            )
        )
        (
            self.bar_note_density_loss_sum,
            self.bar_note_density_match_count,
            self.bar_note_density_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_note_density_loss_sum,
            self.bar_note_density_match_count,
            self.bar_note_density_target_count,
            value=batch_metrics.bar_note_density_loss,
            match_count=batch_metrics.bar_note_density_match_count,
            target_count=batch_metrics.bar_note_density_target_count,
        )
        (
            self.bar_rhythmic_diversity_loss_sum,
            self.bar_rhythmic_diversity_match_count,
            self.bar_rhythmic_diversity_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_rhythmic_diversity_loss_sum,
            self.bar_rhythmic_diversity_match_count,
            self.bar_rhythmic_diversity_target_count,
            value=batch_metrics.bar_rhythmic_diversity_loss,
            match_count=batch_metrics.bar_rhythmic_diversity_match_count,
            target_count=batch_metrics.bar_rhythmic_diversity_target_count,
        )
        (
            self.bar_voice_independence_loss_sum,
            self.bar_voice_independence_match_count,
            self.bar_voice_independence_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_voice_independence_loss_sum,
            self.bar_voice_independence_match_count,
            self.bar_voice_independence_target_count,
            value=batch_metrics.bar_voice_independence_loss,
            match_count=batch_metrics.bar_voice_independence_match_count,
            target_count=batch_metrics.bar_voice_independence_target_count,
        )
        (
            self.bar_uses_accidentals_loss_sum,
            self.bar_uses_accidentals_match_count,
            self.bar_uses_accidentals_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_uses_accidentals_loss_sum,
            self.bar_uses_accidentals_match_count,
            self.bar_uses_accidentals_target_count,
            value=batch_metrics.bar_uses_accidentals_loss,
            match_count=batch_metrics.bar_uses_accidentals_match_count,
            target_count=batch_metrics.bar_uses_accidentals_target_count,
        )
        (
            self.bar_dotted_duration_loss_sum,
            self.bar_dotted_duration_match_count,
            self.bar_dotted_duration_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_dotted_duration_loss_sum,
            self.bar_dotted_duration_match_count,
            self.bar_dotted_duration_target_count,
            value=batch_metrics.bar_dotted_duration_loss,
            match_count=batch_metrics.bar_dotted_duration_match_count,
            target_count=batch_metrics.bar_dotted_duration_target_count,
        )
        (
            self.bar_hand_span_loss_sum,
            self.bar_hand_span_match_count,
            self.bar_hand_span_target_count,
        ) = _add_optional_auxiliary_metric(
            self.bar_hand_span_loss_sum,
            self.bar_hand_span_match_count,
            self.bar_hand_span_target_count,
            value=batch_metrics.bar_hand_span_loss,
            match_count=batch_metrics.bar_hand_span_match_count,
            target_count=batch_metrics.bar_hand_span_target_count,
        )
        (
            self.harmonic_relation_loss_sum,
            self.harmonic_relation_match_count,
            self.harmonic_relation_target_count,
        ) = _add_optional_auxiliary_metric(
            self.harmonic_relation_loss_sum,
            self.harmonic_relation_match_count,
            self.harmonic_relation_target_count,
            value=batch_metrics.harmonic_relation_loss,
            match_count=batch_metrics.harmonic_relation_match_count,
            target_count=batch_metrics.harmonic_relation_target_count,
        )
        self.harmonic_relation_macro_f1_sum = _add_optional_weighted_metric(
            self.harmonic_relation_macro_f1_sum,
            value=batch_metrics.harmonic_relation_macro_f1,
            weight=batch_metrics.harmonic_relation_target_count or 0,
        )
        self.harmonic_relation_target_counts = _add_optional_class_counts(
            self.harmonic_relation_target_counts,
            counts=batch_metrics.harmonic_relation_target_counts,
            class_count=HARMONIC_RELATION_CLASS_COUNT,
        )
        self.harmonic_relation_prediction_counts = _add_optional_class_counts(
            self.harmonic_relation_prediction_counts,
            counts=batch_metrics.harmonic_relation_prediction_counts,
            class_count=HARMONIC_RELATION_CLASS_COUNT,
        )
        self._add_harmonic_plan_metrics(batch_metrics)
        self.harmony_gate_mean_sum, self.harmony_gate_token_count = _add_optional_weighted_loss(
            self.harmony_gate_mean_sum,
            self.harmony_gate_token_count,
            value=batch_metrics.harmony_gate_mean,
            target_count=batch_metrics.harmony_gate_token_count,
        )
        if batch_metrics.validity_penalty_token_count is not None:
            if self.validity_penalty_token_count is None:
                self.validity_penalty_token_count = 0
                self.validity_penalty_loss_sum = 0.0
                self.invalid_probability_mass_sum = 0.0
                self.invalid_target_count = 0

            self.validity_penalty_token_count += batch_metrics.validity_penalty_token_count
            self.validity_penalty_loss_sum = (self.validity_penalty_loss_sum or 0.0) + (
                batch_metrics.validity_penalty_loss or 0.0
            ) * batch_metrics.validity_penalty_token_count
            self.invalid_probability_mass_sum = (self.invalid_probability_mass_sum or 0.0) + (
                batch_metrics.invalid_probability_mass or 0.0
            ) * batch_metrics.validity_penalty_token_count
            self.invalid_target_count = (self.invalid_target_count or 0) + (batch_metrics.invalid_target_count or 0)
        self._add_gradient_norm_metrics(batch_metrics)

    def _add_harmonic_plan_metrics(self, batch_metrics: BatchMetrics) -> None:
        self.harmonic_plan_reconstruction_loss_sum, self.harmonic_plan_reconstruction_target_count = (
            _add_optional_weighted_loss(
                self.harmonic_plan_reconstruction_loss_sum,
                self.harmonic_plan_reconstruction_target_count,
                value=batch_metrics.harmonic_plan_reconstruction_loss,
                target_count=batch_metrics.harmonic_plan_reconstruction_target_count,
            )
        )
        (
            self.harmonic_plan_reconstruction_harmonic_function_match_count,
            self.harmonic_plan_reconstruction_harmonic_function_target_count,
        ) = _add_optional_count_pair(
            self.harmonic_plan_reconstruction_harmonic_function_match_count,
            self.harmonic_plan_reconstruction_harmonic_function_target_count,
            match_count=batch_metrics.harmonic_plan_reconstruction_harmonic_function_match_count,
            target_count=batch_metrics.harmonic_plan_reconstruction_harmonic_function_target_count,
        )
        (
            self.harmonic_plan_reconstruction_root_degree_match_count,
            self.harmonic_plan_reconstruction_root_degree_target_count,
        ) = _add_optional_count_pair(
            self.harmonic_plan_reconstruction_root_degree_match_count,
            self.harmonic_plan_reconstruction_root_degree_target_count,
            match_count=batch_metrics.harmonic_plan_reconstruction_root_degree_match_count,
            target_count=batch_metrics.harmonic_plan_reconstruction_root_degree_target_count,
        )
        (
            self.harmonic_plan_reconstruction_quality_match_count,
            self.harmonic_plan_reconstruction_quality_target_count,
        ) = _add_optional_count_pair(
            self.harmonic_plan_reconstruction_quality_match_count,
            self.harmonic_plan_reconstruction_quality_target_count,
            match_count=batch_metrics.harmonic_plan_reconstruction_quality_match_count,
            target_count=batch_metrics.harmonic_plan_reconstruction_quality_target_count,
        )
        (
            self.harmonic_plan_reconstruction_extension_match_count,
            self.harmonic_plan_reconstruction_extension_target_count,
        ) = _add_optional_count_pair(
            self.harmonic_plan_reconstruction_extension_match_count,
            self.harmonic_plan_reconstruction_extension_target_count,
            match_count=batch_metrics.harmonic_plan_reconstruction_extension_match_count,
            target_count=batch_metrics.harmonic_plan_reconstruction_extension_target_count,
        )
        (
            self.harmonic_plan_reconstruction_cadence_strength_match_count,
            self.harmonic_plan_reconstruction_cadence_strength_target_count,
        ) = _add_optional_count_pair(
            self.harmonic_plan_reconstruction_cadence_strength_match_count,
            self.harmonic_plan_reconstruction_cadence_strength_target_count,
            match_count=batch_metrics.harmonic_plan_reconstruction_cadence_strength_match_count,
            target_count=batch_metrics.harmonic_plan_reconstruction_cadence_strength_target_count,
        )
        (
            self.harmonic_plan_contrastive_loss_sum,
            self.harmonic_plan_contrastive_match_count,
            self.harmonic_plan_contrastive_target_count,
        ) = _add_optional_auxiliary_metric(
            self.harmonic_plan_contrastive_loss_sum,
            self.harmonic_plan_contrastive_match_count,
            self.harmonic_plan_contrastive_target_count,
            value=batch_metrics.harmonic_plan_contrastive_loss,
            match_count=batch_metrics.harmonic_plan_contrastive_match_count,
            target_count=batch_metrics.harmonic_plan_contrastive_target_count,
        )
        self.harmonic_plan_contrastive_positive_similarity_sum = _add_optional_weighted_metric(
            self.harmonic_plan_contrastive_positive_similarity_sum,
            value=batch_metrics.harmonic_plan_contrastive_positive_similarity,
            weight=batch_metrics.harmonic_plan_contrastive_target_count or 0,
        )
        self.harmonic_plan_contrastive_negative_similarity_sum = _add_optional_weighted_metric(
            self.harmonic_plan_contrastive_negative_similarity_sum,
            value=batch_metrics.harmonic_plan_contrastive_negative_similarity,
            weight=batch_metrics.harmonic_plan_contrastive_target_count or 0,
        )

    def _add_gradient_norm_metrics(self, batch_metrics: BatchMetrics) -> None:
        self.cnn_gradient_norm_sum = _add_optional_weighted_metric(
            self.cnn_gradient_norm_sum,
            value=batch_metrics.cnn_gradient_norm,
            weight=batch_metrics.token_count,
        )
        self.gru_gradient_norm_sum = _add_optional_weighted_metric(
            self.gru_gradient_norm_sum,
            value=batch_metrics.gru_gradient_norm,
            weight=batch_metrics.token_count,
        )
        self.transformer_gradient_norm_sum = _add_optional_weighted_metric(
            self.transformer_gradient_norm_sum,
            value=batch_metrics.transformer_gradient_norm,
            weight=batch_metrics.token_count,
        )

    def to_epoch_split_metrics(self) -> EpochSplitMetrics:
        if self.token_count == 0:
            raise ValueError("cannot calculate metrics without tokens")

        loss = self.loss_sum / self.token_count
        return EpochSplitMetrics(
            loss=loss,
            perplexity=exp(loss),
            token_accuracy=self.exact_match_count / self.token_count,
            token_kind_accuracy=(
                None if self.token_kind_match_count is None else self.token_kind_match_count / self.token_count
            ),
            event_kind_loss=_weighted_optional_average(
                self.event_kind_loss_sum,
                weight=self.event_kind_loss_target_count,
            ),
            duration_loss=_weighted_optional_average(self.duration_loss_sum, weight=self.duration_loss_target_count),
            degree_loss=_weighted_optional_average(self.degree_loss_sum, weight=self.degree_loss_target_count),
            accidental_loss=_weighted_optional_average(
                self.accidental_loss_sum,
                weight=self.accidental_loss_target_count,
            ),
            octave_offset_loss=_weighted_optional_average(
                self.octave_offset_loss_sum,
                weight=self.octave_offset_loss_target_count,
            ),
            hand_loss=_weighted_optional_average(self.hand_loss_sum, weight=self.hand_loss_target_count),
            duration_accuracy=_optional_rate(self.duration_match_count, target_count=self.duration_target_count),
            degree_accuracy=_optional_rate(self.degree_match_count, target_count=self.degree_target_count),
            accidental_accuracy=_optional_rate(
                self.accidental_match_count,
                target_count=self.accidental_target_count,
            ),
            octave_offset_accuracy=_optional_rate(
                self.octave_offset_match_count,
                target_count=self.octave_offset_target_count,
            ),
            hand_accuracy=_optional_rate(self.hand_match_count, target_count=self.hand_target_count),
            musical_auxiliary_loss=_weighted_optional_average(
                self.musical_auxiliary_loss_sum,
                weight=self.musical_auxiliary_target_count,
            ),
            note_density_loss=_weighted_optional_average(
                self.note_density_loss_sum,
                weight=self.note_density_target_count,
            ),
            note_density_accuracy=_optional_rate(
                self.note_density_match_count,
                target_count=self.note_density_target_count,
            ),
            rhythmic_diversity_loss=_weighted_optional_average(
                self.rhythmic_diversity_loss_sum,
                weight=self.rhythmic_diversity_target_count,
            ),
            rhythmic_diversity_accuracy=_optional_rate(
                self.rhythmic_diversity_match_count,
                target_count=self.rhythmic_diversity_target_count,
            ),
            voice_independence_loss=_weighted_optional_average(
                self.voice_independence_loss_sum,
                weight=self.voice_independence_target_count,
            ),
            voice_independence_accuracy=_optional_rate(
                self.voice_independence_match_count,
                target_count=self.voice_independence_target_count,
            ),
            uses_accidentals_loss=_weighted_optional_average(
                self.uses_accidentals_loss_sum,
                weight=self.uses_accidentals_target_count,
            ),
            uses_accidentals_accuracy=_optional_rate(
                self.uses_accidentals_match_count,
                target_count=self.uses_accidentals_target_count,
            ),
            dotted_duration_loss=_weighted_optional_average(
                self.dotted_duration_loss_sum,
                weight=self.dotted_duration_target_count,
            ),
            dotted_duration_accuracy=_optional_rate(
                self.dotted_duration_match_count,
                target_count=self.dotted_duration_target_count,
            ),
            hand_span_loss=_weighted_optional_average(self.hand_span_loss_sum, weight=self.hand_span_target_count),
            hand_span_accuracy=_optional_rate(self.hand_span_match_count, target_count=self.hand_span_target_count),
            bar_note_density_loss=_weighted_optional_average(
                self.bar_note_density_loss_sum,
                weight=self.bar_note_density_target_count,
            ),
            bar_note_density_accuracy=_optional_rate(
                self.bar_note_density_match_count,
                target_count=self.bar_note_density_target_count,
            ),
            bar_rhythmic_diversity_loss=_weighted_optional_average(
                self.bar_rhythmic_diversity_loss_sum,
                weight=self.bar_rhythmic_diversity_target_count,
            ),
            bar_rhythmic_diversity_accuracy=_optional_rate(
                self.bar_rhythmic_diversity_match_count,
                target_count=self.bar_rhythmic_diversity_target_count,
            ),
            bar_voice_independence_loss=_weighted_optional_average(
                self.bar_voice_independence_loss_sum,
                weight=self.bar_voice_independence_target_count,
            ),
            bar_voice_independence_accuracy=_optional_rate(
                self.bar_voice_independence_match_count,
                target_count=self.bar_voice_independence_target_count,
            ),
            bar_uses_accidentals_loss=_weighted_optional_average(
                self.bar_uses_accidentals_loss_sum,
                weight=self.bar_uses_accidentals_target_count,
            ),
            bar_uses_accidentals_accuracy=_optional_rate(
                self.bar_uses_accidentals_match_count,
                target_count=self.bar_uses_accidentals_target_count,
            ),
            bar_dotted_duration_loss=_weighted_optional_average(
                self.bar_dotted_duration_loss_sum,
                weight=self.bar_dotted_duration_target_count,
            ),
            bar_dotted_duration_accuracy=_optional_rate(
                self.bar_dotted_duration_match_count,
                target_count=self.bar_dotted_duration_target_count,
            ),
            bar_hand_span_loss=_weighted_optional_average(
                self.bar_hand_span_loss_sum,
                weight=self.bar_hand_span_target_count,
            ),
            bar_hand_span_accuracy=_optional_rate(
                self.bar_hand_span_match_count,
                target_count=self.bar_hand_span_target_count,
            ),
            harmonic_relation_loss=_weighted_optional_average(
                self.harmonic_relation_loss_sum,
                weight=self.harmonic_relation_target_count,
            ),
            harmonic_relation_accuracy=_optional_rate(
                self.harmonic_relation_match_count,
                target_count=self.harmonic_relation_target_count,
            ),
            harmonic_relation_macro_f1=_weighted_optional_average(
                self.harmonic_relation_macro_f1_sum,
                weight=self.harmonic_relation_target_count,
            ),
            harmonic_relation_target_distribution=_optional_distribution(self.harmonic_relation_target_counts),
            harmonic_relation_prediction_distribution=_optional_distribution(self.harmonic_relation_prediction_counts),
            harmonic_plan_reconstruction_loss=_weighted_optional_average(
                self.harmonic_plan_reconstruction_loss_sum,
                weight=self.harmonic_plan_reconstruction_target_count,
            ),
            harmonic_plan_reconstruction_harmonic_function_accuracy=_optional_rate(
                self.harmonic_plan_reconstruction_harmonic_function_match_count,
                target_count=self.harmonic_plan_reconstruction_harmonic_function_target_count,
            ),
            harmonic_plan_reconstruction_root_degree_accuracy=_optional_rate(
                self.harmonic_plan_reconstruction_root_degree_match_count,
                target_count=self.harmonic_plan_reconstruction_root_degree_target_count,
            ),
            harmonic_plan_reconstruction_quality_accuracy=_optional_rate(
                self.harmonic_plan_reconstruction_quality_match_count,
                target_count=self.harmonic_plan_reconstruction_quality_target_count,
            ),
            harmonic_plan_reconstruction_extension_accuracy=_optional_rate(
                self.harmonic_plan_reconstruction_extension_match_count,
                target_count=self.harmonic_plan_reconstruction_extension_target_count,
            ),
            harmonic_plan_reconstruction_cadence_strength_accuracy=_optional_rate(
                self.harmonic_plan_reconstruction_cadence_strength_match_count,
                target_count=self.harmonic_plan_reconstruction_cadence_strength_target_count,
            ),
            harmonic_plan_contrastive_loss=_weighted_optional_average(
                self.harmonic_plan_contrastive_loss_sum,
                weight=self.harmonic_plan_contrastive_target_count,
            ),
            harmonic_plan_contrastive_accuracy=_optional_rate(
                self.harmonic_plan_contrastive_match_count,
                target_count=self.harmonic_plan_contrastive_target_count,
            ),
            harmonic_plan_contrastive_positive_similarity=_weighted_optional_average(
                self.harmonic_plan_contrastive_positive_similarity_sum,
                weight=self.harmonic_plan_contrastive_target_count,
            ),
            harmonic_plan_contrastive_negative_similarity=_weighted_optional_average(
                self.harmonic_plan_contrastive_negative_similarity_sum,
                weight=self.harmonic_plan_contrastive_target_count,
            ),
            harmony_gate_mean=_weighted_optional_average(
                self.harmony_gate_mean_sum,
                weight=self.harmony_gate_token_count,
            ),
            validity_penalty_loss=_optional_validity_average(
                self.validity_penalty_loss_sum,
                token_count=self.validity_penalty_token_count,
            ),
            invalid_probability_mass=_optional_validity_average(
                self.invalid_probability_mass_sum,
                token_count=self.validity_penalty_token_count,
            ),
            invalid_target_rate=(
                None if self.invalid_target_count is None else self.invalid_target_count / self.token_count
            ),
            cnn_gradient_norm=_weighted_optional_average(self.cnn_gradient_norm_sum, weight=self.token_count),
            gru_gradient_norm=_weighted_optional_average(self.gru_gradient_norm_sum, weight=self.token_count),
            transformer_gradient_norm=_weighted_optional_average(
                self.transformer_gradient_norm_sum,
                weight=self.token_count,
            ),
        )


def _add_optional_weighted_metric(current: float | None, *, value: float | None, weight: int) -> float | None:
    if value is None:
        return current

    return (current or 0.0) + value * weight


def _add_optional_weighted_loss(
    current_sum: float | None,
    current_target_count: int | None,
    *,
    value: float | None,
    target_count: int | None,
) -> tuple[float | None, int | None]:
    if value is None or target_count is None:
        return current_sum, current_target_count

    return (current_sum or 0.0) + value * target_count, (current_target_count or 0) + target_count


def _add_optional_count_pair(
    current_match_count: int | None,
    current_target_count: int | None,
    *,
    match_count: int | None,
    target_count: int | None,
) -> tuple[int | None, int | None]:
    if match_count is None or target_count is None:
        return current_match_count, current_target_count

    return (current_match_count or 0) + match_count, (current_target_count or 0) + target_count


def _add_optional_auxiliary_metric(
    current_loss_sum: float | None,
    current_match_count: int | None,
    current_target_count: int | None,
    *,
    value: float | None,
    match_count: int | None,
    target_count: int | None,
) -> tuple[float | None, int | None, int | None]:
    if value is None or match_count is None or target_count is None:
        return current_loss_sum, current_match_count, current_target_count

    return (
        (current_loss_sum or 0.0) + value * target_count,
        (current_match_count or 0) + match_count,
        (current_target_count or 0) + target_count,
    )


def _add_optional_class_counts(
    current_counts: list[int] | None,
    *,
    counts: tuple[int, ...] | None,
    class_count: int,
) -> list[int] | None:
    if counts is None:
        return current_counts

    if len(counts) != class_count:
        raise ValueError(f"class count vector must have length {class_count}")

    output = current_counts or [0] * class_count
    for class_index, count in enumerate(counts):
        output[class_index] += count

    return output


def _optional_distribution(counts: list[int] | None) -> tuple[float, ...] | None:
    if counts is None:
        return None

    total = sum(counts)
    if total == 0:
        return None

    return tuple(count / total for count in counts)


def _optional_rate(
    match_count: int | None,
    *,
    target_count: int | None,
) -> float | None:
    if match_count is None or target_count is None or target_count == 0:
        return None

    return match_count / target_count


def _weighted_optional_average(
    value: float | None,
    *,
    weight: int | None,
) -> float | None:
    if value is None or weight is None or weight == 0:
        return None

    return value / weight


def _optional_validity_average(
    value: float | None,
    *,
    token_count: int | None,
) -> float | None:
    if value is None or token_count is None or token_count == 0:
        return None

    return value / token_count
