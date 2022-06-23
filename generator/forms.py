from django import forms


class SettingsForm(forms.Form):
    time_signature_numerator = forms.IntegerField(label='Time signature', min_value=1, max_value=32, initial=4)
    time_signature_denominator = forms.IntegerField(label='Time signature', min_value=1, max_value=32, initial=4)
    groups = forms.IntegerField(label='Groups', min_value=1, max_value=4, initial=1)
    measures = forms.IntegerField(label='Measures', min_value=1, max_value=8, initial=2)

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

    left_quarter_phrase = forms.BooleanField(label='\U0001d15f\U0001d13d', required=False)
    right_quarter_phrase = forms.BooleanField(label='\U0001d13d\U0001d15f', required=False)
    left_eighth_phrase = forms.BooleanField(label='\U0001d160\U0001d13e', required=False)
    right_eighth_phrase = forms.BooleanField(label='\U0001d13e\U0001d160', required=False)
    left_sixteenth_phrase = forms.BooleanField(label='\U0001d161\U0001d13f', required=False)
    right_sixteenth_phrase = forms.BooleanField(label='\U0001d13f\U0001d161', required=False)

    custom_phrases = forms.CharField(label='', required=False)
