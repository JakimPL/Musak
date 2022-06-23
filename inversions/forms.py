from django import forms

from config.defaults import TEMPO, MIN_TEMPO, MAX_TEMPO


class SettingsForm(forms.Form):
    sequential = forms.BooleanField(label='Sequential', required=False)
    tempo = forms.IntegerField(label='Tempo', min_value=MIN_TEMPO, max_value=MAX_TEMPO, initial=TEMPO)
