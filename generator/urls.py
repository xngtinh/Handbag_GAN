from django.urls import path
from . import views
from . import views_tote
from . import views_clutch
from . import views_satchel

urlpatterns = [
    path('', views.home, name='gen-home'),
    # path('images/',views.images, name='gen-images'),
    # path('images/tote/',views_tote.images, name='gen-tote'),
    # path('images/clutch/',views_clutch.images, name='gen-clutch'),
    # path('images/satchel/',views_satchel.images, name='gen-satchel'),
    # path('about/', views.about, name='gen-about'),
    # path('aboutproject/', views.aboutproject, name='gen-aboutproject'),
    # path('future/', views.future, name='gen-future'),
    # path('aboutdevelopers/',views.aboutdevelopers, name='gen-aboutdevelopers')
]