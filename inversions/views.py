from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render

from inversions.forms import SettingsForm
from inversions.models import ChordInversionModel
from shared.dict import get_key


def submit_inversion(request) -> JsonResponse:
    if 'submit' in request.POST:
        parameters = request.POST['submit']
        data = parse_qs(parameters)

        options = {key: True for key in data if get_key(data, key) == 'on'}
        chord_inversion_model = ChordInversionModel(sequential=options.get('sequential', False))
        uuid = chord_inversion_model.generate()
        return JsonResponse({
            'directory': uuid,
            'filename': 'chord.mp3'
        })
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    form = SettingsForm(request)
    response = {'form': form}
    return render(request, 'inversions.html', response)
