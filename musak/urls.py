from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve

from intervals.views import submit_interval
from inversions.views import submit_inversion
from musak import settings
from rhythm.views import submit_rhythm

urlpatterns = [
    path('', include('start.urls')),
    path('start/', include('start.urls')),
    path('inversions/', include('inversions.urls')),
    path('intervals/', include('intervals.urls')),
    path('rhythm/', include('rhythm.urls')),
    path('admin/', admin.site.urls),
    path('submit_inversion/', submit_inversion, name='submit_inversion'),
    path('submit_interval/', submit_interval, name='submit_interval'),
    path('submit_rhythm/', submit_rhythm, name='submit_rhythm'),
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_URL}),
    re_path(r'^temp/(?P<path>.*)$', serve, {'document_root': settings.TEMP_URL})
]

urlpatterns.extend(static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT))
