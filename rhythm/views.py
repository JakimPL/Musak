from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render
from rhygen.modules.exceptions import NoteNotSupportedError

from rhythm.forms import SettingsForm
from rhythm.services import RhygenService, get_settings, default_settings


def generate_content(request, form: SettingsForm, rhygen: RhygenService):
    response = {'form': form} | get_response(rhygen)
    return render(request, 'rhythm.html', response)


def get_response(rhygen: RhygenService) -> dict[str, str]:
    response = {}
    if rhygen.score is not None:
        response['score'] = str(rhygen.score)
        response['directory'] = f'../{rhygen.uuid}'
        response['image_source'] = f'../{rhygen.image}'
        response['audio_source'] = f'../{rhygen.audio}'

    if rhygen.exception is not None:
        response['exception'] = str(rhygen.exception)

    if rhygen.time_signature_error:
        response['time_signature_error'] = rhygen.time_signature_error

    return response


def submit_rhythm(request) -> JsonResponse:
    if 'submit' in request.POST:
        parameters = request.POST['submit']
        data = parse_qs(parameters)

        try:
            rhygen = RhygenService(get_settings(data))
            response = JsonResponse(get_response(rhygen))
        except NoteNotSupportedError:
            response = JsonResponse({'error_message': 'unsupported note'}, status=400)

        return response
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    if request.method == 'POST':
        form = SettingsForm(request.POST)
    else:
        form = SettingsForm(default_settings(form=True))

    settings = get_settings(form.data) if request.method == 'POST' and form.is_valid() else default_settings()
    rhygen = RhygenService(settings)
    return generate_content(request, form, rhygen)
