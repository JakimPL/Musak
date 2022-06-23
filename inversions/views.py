from django.http import JsonResponse
from django.shortcuts import render

from inversions.models import ChordInversionModel

chord_inversion_model = ChordInversionModel()


def submit_inversion(request) -> JsonResponse:
    if 'submit' in request.POST:
        uuid = chord_inversion_model.generate()
        return JsonResponse({'directory': uuid})
    else:
        return JsonResponse({'error_message': 'invalid request'}, status=400)


def index(request):
    return render(request, 'inversions.html')
