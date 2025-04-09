from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # or any view you want as default
]
