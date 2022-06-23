from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from musak import settings
from rhythm.views import submit_rhythm
from inversions.views import submit_inversion

urlpatterns = [
    path('rhythm/', include('rhythm.urls')),
    path('inversions/', include('inversions.urls')),
    path('admin/', admin.site.urls),
    path('submit_inversion/', submit_inversion, name='submit_inversion'),
    path('submit_rhythm/', submit_rhythm, name='submit_rhythm')
]

urlpatterns.extend(static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT))
