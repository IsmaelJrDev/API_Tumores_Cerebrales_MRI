from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # Esta línea conecta la raíz ('') con tu app de diagnóstico
    path('', include('diagnosis.urls')), 
]

# Esto es OBLIGATORIO para que se vean las imágenes que subes en modo DEBUG
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)