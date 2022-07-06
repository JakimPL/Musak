from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render

from inversions.forms import SettingsForm
from inversions.models import ChordInversionModel
from inversions.services import default_settings, get_settings, get_chords_definitions

chords_definitions = get_chords_definitions()


def submit_inversion(request) -> JsonResponse:
    if 'submit' in request.POST:
        parameters = request.POST['submit']
        data = parse_qs(parameters)

        chord_inversion_model = ChordInversionModel(get_settings(data, chords_definitions))
        uuid = chord_inversion_model.generate()
        chord_types = list(chord_inversion_model.chords.keys())
        inversions_numbers = {key: len(value) for key, value in chord_inversion_model.inversions.items()}

        return JsonResponse({
            'directory': uuid,
            'chord_types': chord_types,
            'inversions_numbers': inversions_numbers,
            'chord_info': 'chord.json',
            'audio_source': 'chord.mp3',
            'image_source': 'chord.png'
        })
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    if request.method == 'POST':
        form = SettingsForm(data=request.POST, chords_definitions=chords_definitions)
    else:
        form = SettingsForm(data=default_settings(form=True), chords_definitions=chords_definitions)

    response = {'form': form}
    return render(request, 'inversions.html', response)
