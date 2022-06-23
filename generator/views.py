import re
from typing import Dict, Any, Union
from urllib.parse import parse_qs

from django.http import JsonResponse
from django.shortcuts import render
from rhygen.modules.exceptions import NoteNotSupportedError

from generator.forms import SettingsForm
from generator.services import RhygenService, settings_map


def get_key(data: Dict[str, Any], key: str) -> Any:
    element = data[key]
    if isinstance(element, list) and len(element) == 1:
        return element[0]

    return element


def default_settings(form: bool = False) -> Dict[str, Any]:
    settings = {
        'groups': 1,
        'measures': 2,
        'time_signature_numerator': 4,
        'time_signature_denominator': 4,
        'half_note': 'on',
        'half_rest': 'on',
        'quarter_note': 'on'
    }

    return settings if form else get_settings(settings)


def parse_custom_phrase(raw_phrase: str) -> list:
    elements = raw_phrase.split(',')
    notes = []

    for element in elements:
        if '(' in element or ')' in element:
            if re.match(r"(-)?\(\d+:\d+\)", element):
                raw_pair = element.replace('(', '').replace(')', '').split(':')
                note = int(raw_pair[0]), int(raw_pair[1])
            else:
                raise ValueError(f'invalid note element {element}')
        else:
            note = int(element)

        notes.append(note)

    return notes


def parse_custom_phrases(phrases_string: str) -> list:
    string = phrases_string.strip().replace(' ', '')

    count = 0
    elements = []
    raw_phrase = ''
    for char in string:
        if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '[', ']', '(', ')', '-', ':']:
            raise ValueError(f'unexpected symbol {char}')
        bracket = True
        if char == '[':
            count += 1
        elif char == ']':
            count -= 1
            if raw_phrase:
                elements.append(raw_phrase)

            raw_phrase = ''
        else:
            bracket = False
            if count != 0:
                raw_phrase += char

        if count < 0:
            raise ValueError('unexpected closing bracket')
        elif count > 1:
            raise ValueError('unexpected nested expression')
        elif count == 0 and char != ',' and not bracket:
            raise ValueError(f'unexpected symbol {char} outside the expression')

    if count != 0:
        raise ValueError('unbalanced square brackets')
    return [parse_custom_phrase(raw_phrase) for raw_phrase in elements]


def get_settings(data: Dict[str, Union[str, list]]) -> Dict[str, Any]:
    settings = {
        'notes': [],
        'phrases': []
    }

    try:
        settings['groups'] = int(get_key(data, 'groups'))
        settings['measures'] = int(get_key(data, 'measures'))
        settings['time_signature'] = (
            int(get_key(data, 'time_signature_numerator')),
            int(get_key(data, 'time_signature_denominator'))
        )
        options = {key: get_key(data, key) for key in data if get_key(data, key) == 'on'}
        for key, value in options.items():
            if '_phrase' in key:
                settings['phrases'].append(settings_map[key])
            else:
                settings['notes'].append(settings_map[key])

        settings['phrases'] += parse_custom_phrases(get_key(data, 'custom_phrases'))
    except KeyError:
        pass
    except TypeError:
        pass

    return settings


def generate_content(request, form: SettingsForm, rhygen: RhygenService):
    response = {'form': form} | get_response(rhygen)
    return render(request, 'generator.html', response)


def get_response(rhygen: RhygenService) -> Dict[str, str]:
    response = {}
    if rhygen.score is not None:
        response['score'] = str(rhygen.score)
        response['image_source'] = f'../{rhygen.image}'

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
        form = SettingsForm(request)
        settings = get_settings(form.data) if request.method == 'POST' and form.is_valid() else default_settings()
    else:
        form = SettingsForm(default_settings(form=True))
        settings = get_settings(form.data)

    rhygen = RhygenService(settings)
    return generate_content(request, form, rhygen)
