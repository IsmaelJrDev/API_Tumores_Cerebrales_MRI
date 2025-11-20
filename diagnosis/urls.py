from django.urls import path
from . import views

urlpatterns = [
    # Cuando alguien entre a la raíz, ejecuta la vista 'home'
    path('', views.home, name='home'),
]