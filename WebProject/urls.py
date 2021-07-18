"""WebProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from generator import views as generator_views
from generator import views_tote as generator_views_tote
from generator import views_clutch as generator_views_clutch
from generator import views_satchel as generator_views_satchel

urlpatterns = [
    path('admin/', admin.site.urls),
    path('balo/', generator_views.images, name="gen-balo"),
    path('tote/', generator_views_tote.images, name="gen-tote"),
    path('clutch/', generator_views_clutch.images, name="gen-clutch"),
    path('satchel/', generator_views_satchel.images, name="gen-satchel"),
    path('', include('generator.urls')),
]
