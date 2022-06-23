from django import forms

from config.defaults import (
    TEMPO,
    MIN_TEMPO,
    MAX_TEMPO,
    LOWEST_NOTE,
    MIN_LOWEST_NOTE,
    MAX_LOWEST_NOTE,
    HIGHEST_NOTE,
    MIN_HIGHEST_NOTE,
    MAX_HIGHEST_NOTE
)


class SettingsForm(forms.Form):
    sequential = forms.BooleanField(label='Sequential', required=False)
    tempo = forms.IntegerField(label='Tempo', min_value=MIN_TEMPO, max_value=MAX_TEMPO, initial=TEMPO)
    lowest_note = forms.IntegerField(min_value=MIN_LOWEST_NOTE, max_value=MAX_LOWEST_NOTE, initial=LOWEST_NOTE)
    highest_note = forms.IntegerField(min_value=MIN_HIGHEST_NOTE, max_value=MAX_HIGHEST_NOTE, initial=HIGHEST_NOTE)

    def __init__(self, chords_definitions: dict[str, list[int]], *args, **kwargs):
        super(SettingsForm, self).__init__(*args, **kwargs)

        self.chords = []
        for key, value in chords_definitions.items():
            widget = forms.BooleanField(label=key, required=False)
            self.fields[f'chord_{key}'] = widget
            self.chords.append(widget)

    def get_chords(self):
        for field_name in self.fields:
            if field_name.startswith('chord_'):
                yield self[field_name]
