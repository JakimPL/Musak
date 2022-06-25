from django import forms

from config.defaults import (
    TEMPO,
    MIN_TEMPO,
    MAX_TEMPO,
    GROUPS,
    MIN_GROUPS,
    MAX_GROUPS,
    MEASURES,
    MIN_MEASURES,
    MAX_MEASURES,
    TIME_SIGNATURE_NUMERATOR,
    MIN_TIME_SIGNATURE_NUMERATOR,
    MAX_TIME_SIGNATURE_NUMERATOR,
    TIME_SIGNATURE_DENOMINATOR,
    MIN_TIME_SIGNATURE_DENOMINATOR,
    MAX_TIME_SIGNATURE_DENOMINATOR
)


class SettingsForm(forms.Form):
    tempo = forms.IntegerField(label='Tempo', min_value=MIN_TEMPO, max_value=MAX_TEMPO, initial=TEMPO)
    time_signature_numerator = forms.IntegerField(label='Time signature', min_value=MIN_TIME_SIGNATURE_NUMERATOR, max_value=MAX_TIME_SIGNATURE_NUMERATOR, initial=TIME_SIGNATURE_NUMERATOR)
    time_signature_denominator = forms.IntegerField(label='Time signature', min_value=MIN_TIME_SIGNATURE_DENOMINATOR, max_value=MAX_TIME_SIGNATURE_DENOMINATOR, initial=TIME_SIGNATURE_DENOMINATOR)
    groups = forms.IntegerField(label='Groups', min_value=MIN_GROUPS, max_value=MAX_GROUPS, initial=GROUPS)
    measures = forms.IntegerField(label='Measures', min_value=MIN_MEASURES, max_value=MAX_MEASURES, initial=MEASURES)

    whole_note = forms.BooleanField(label='\U0001d15d', required=False)
    half_note = forms.BooleanField(label='\U0001d15e', required=False)
    quarter_note = forms.BooleanField(label='\U0001d15f', required=False)
    eighth_note = forms.BooleanField(label='\U0001d160', required=False)
    sixteenth_note = forms.BooleanField(label='\U0001d160', required=False)
    thirty_second_note = forms.BooleanField(label='\U0001d161', required=False)

    whole_rest = forms.BooleanField(label='\U0001d13b', required=False)
    half_rest = forms.BooleanField(label='\U0001d13c', required=False)
    quarter_rest = forms.BooleanField(label='\U0001d13d', required=False)
    eighth_rest = forms.BooleanField(label='\U0001d13e', required=False)
    sixteenth_rest = forms.BooleanField(label='\U0001d13f', required=False)
    thirty_second_rest = forms.BooleanField(label='\U0001d140', required=False)

    dotted_half_note = forms.BooleanField(label='\U0001d15e.', required=False)
    dotted_quarter_note = forms.BooleanField(label='\U0001d15f.', required=False)
    dotted_eighth_note = forms.BooleanField(label='\U0001d160.', required=False)
    dotted_sixteenth_note = forms.BooleanField(label='\U0001d160.', required=False)

    two_quarter_notes_phrase = forms.BooleanField(label='\U0001d15f\U0001d15f', required=False)
    two_eighth_notes_phrase = forms.BooleanField(label='\U0000266b', required=False)
    four_eighth_notes_phrase = forms.BooleanField(label='\U0000266b\U0000266b', required=False)
    two_sixteenth_notes_phrase = forms.BooleanField(label='\U0000266c', required=False)
    four_sixteenth_notes_phrase = forms.BooleanField(label='\U0000266c\U0000266c', required=False)
    eight_sixteenth_notes_phrase = forms.BooleanField(label='\U0000266c\U0000266c\U0000266c\U0000266c', required=False)

    left_quarter_phrase = forms.BooleanField(label='\U0001d15f\u00A0\U0001d13d', required=False)
    right_quarter_phrase = forms.BooleanField(label='\U0001d13d\u00A0\U0001d15f', required=False)
    left_eighth_phrase = forms.BooleanField(label='\U0001d160\u00A0\U0001d13e', required=False)
    right_eighth_phrase = forms.BooleanField(label='\U0001d13e\u00A0\U0001d160', required=False)
    left_sixteenth_phrase = forms.BooleanField(label='\U0001d161\u00A0\U0001d13f', required=False)
    right_sixteenth_phrase = forms.BooleanField(label='\U0001d13f\u00A0\U0001d161', required=False)

    custom_phrases = forms.CharField(label='', required=False)
