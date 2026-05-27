from dataclasses import dataclass

from musak_model.data.schema import Segment
from musak_model.synthetic.harmony.decoding.candidates import spellable_candidates
from musak_model.synthetic.harmony.decoding.config import ChordDecoderConfig
from musak_model.synthetic.harmony.decoding.schema import ChordWindow
from musak_model.synthetic.harmony.decoding.viterbi import viterbi_decode
from musak_model.synthetic.harmony.decoding.windows import sounding_windows
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import DurationVocabulary


@dataclass(frozen=True)
class ViterbiChordDecoder:
    config: ChordDecoderConfig

    def decode(
        self,
        segment: Segment,
        *,
        duration_vocabulary: DurationVocabulary,
        vocabulary: ChordVocabularyConfig,
    ) -> tuple[ChordWindow, ...]:
        windows = sounding_windows(
            segment,
            duration_vocabulary=duration_vocabulary,
            resolution=self.config.resolution,
        )
        candidates = spellable_candidates(vocabulary, scale_type=segment.scale_type)
        if not windows or not candidates:
            return ()

        chords = viterbi_decode(
            windows,
            candidates=candidates,
            self_transition_bias=self.config.self_transition_bias,
            non_chord_penalty=self.config.non_chord_penalty,
        )
        return tuple(
            ChordWindow(start=window.start, end=window.end, chord=chord)
            for window, chord in zip(windows, chords, strict=True)
        )
