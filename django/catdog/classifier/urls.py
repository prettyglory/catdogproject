from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('classify/', views.classify_image, name='classify'),
]
