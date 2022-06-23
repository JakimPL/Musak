from django import forms


class SettingsForm(forms.Form):
    sequential = forms.BooleanField(label='Sequential', required=False)
