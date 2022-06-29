from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render

from intervals.forms import SettingsForm
from intervals.models import IntervalModel
from intervals.services import default_settings, get_settings, get_intervals_definitions

intervals_definitions = get_intervals_definitions()


def submit_interval(request) -> JsonResponse:
    if 'submit' in request.POST:
        parameters = request.POST['submit']
        data = parse_qs(parameters)

        interval_model = IntervalModel(get_settings(data, intervals_definitions))
        uuid = interval_model.generate()
        return JsonResponse({
            'directory': uuid,
            'interval_info': 'interval.json',
            'audio_source': 'interval.mp3',
            'image_source': 'interval.png',
            'intervals': interval_model.intervals
        })
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    if request.method == 'POST':
        form = SettingsForm(data=request.POST, intervals_definitions=intervals_definitions)
    else:
        form = SettingsForm(data=default_settings(form=True), intervals_definitions=intervals_definitions)

    response = {'form': form}
    return render(request, 'intervals.html', response)
