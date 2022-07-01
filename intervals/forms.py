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

    def __init__(self, intervals_definitions: dict[str, int], *args, **kwargs):
        super(SettingsForm, self).__init__(*args, **kwargs)

        self.intervals = []
        for key, value in intervals_definitions.items():
            widget = forms.BooleanField(label=key.replace('_', ' '), required=False)
            self.fields[f'interval_{key}'] = widget
            self.intervals.append(widget)

    def get_intervals(self):
        for field_name in self.fields:
            if field_name.startswith('interval_'):
                yield self[field_name]
