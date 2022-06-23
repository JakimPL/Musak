from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render

from inversions.forms import SettingsForm
from inversions.models import ChordInversionModel
from inversions.services import default_settings, get_settings


def submit_inversion(request) -> JsonResponse:
    if 'submit' in request.POST:
        parameters = request.POST['submit']
        data = parse_qs(parameters)

        chord_inversion_model = ChordInversionModel(get_settings(data))
        uuid = chord_inversion_model.generate()
        return JsonResponse({'directory': uuid, 'filename': 'chord.mp3'})
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    if request.method == 'POST':
        form = SettingsForm(request)
    else:
        form = SettingsForm(default_settings(form=True))

    response = {'form': form}
    return render(request, 'inversions.html', response)
